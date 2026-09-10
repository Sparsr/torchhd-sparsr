// PyTorch PrivateUse1 backend for Sparsr ("sparsr" device).
//
// This file registers the device and translates between PyTorch tensors and
// hypervectors. It computes nothing itself: every HDC operation is a call into
// libsparsr_hdc, which owns the algorithms and the device kernels that run them.
// There used to be a second set of kernels here -- bind and bundle,
// written independently of the HDC library's -- and two
// implementations of one operation drift apart with nothing to catch it.
//
// So the layering is one way: torchhd is a helper library for HDC, the HDC
// operations are libsparsr_hdc's, and this file owns the PyTorch side alone.
//
// WHERE A TENSOR ACTUALLY LIVES, AND WHY THAT CHANGED
//
// A "sparsr" tensor holds its bits in host memory, and each operation sends its
// operands to the device and reads the result back. It used to be resident: a
// tensor's storage pointer encoded a CMEM row, and the bits stayed on the device
// between operations.
//
// That could not survive this change, and it should not have. libsparsr_hdc
// reserves CMEM rows 0 to 31 -- every row there is -- from hdc_init() onwards,
// and nothing on a Sparsr device arbitrates who owns a row. Residency here meant
// two libraries writing the same rows with no error on either side, which
// the HDC library's own device memory layout names as the exact
// collision it cannot prevent. One owner of CMEM is the only arrangement that
// works today. A host-side memory manager would let residency come back; it is
// planned and not built.
//
// Only single hypervectors (exactly 4096 bool elements) are supported per tensor.

#include <torch/extension.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <tuple>

#include "sparsr_hdc.h"

namespace {

constexpr int64_t kHypervectorBits = HDC_HYPERVECTOR_BITS;

// PyTorch stores one full byte per bool element, so a hypervector tensor is
// kHypervectorBits bytes of storage -- not the HDC_HYPERVECTOR_BYTES bit-packed
// form libsparsr_hdc takes. The two conversions below are the only place that
// difference exists.
constexpr int64_t kTensorBytes = kHypervectorBits;

// =====================================================================
// Hypervector conversion
// =====================================================================

// Bit order is libsparsr_hdc's, reached through its own accessors rather than
// re-derived here: this file must not carry a second opinion about which bit of
// which byte position i is. Every operation the library exposes is a bitwise one,
// so the ordering only has to be consistent -- which is exactly why getting it
// from one place matters.

hdc_hypervector to_hypervector(const bool* elements) {
  hdc_hypervector vector;
  hdc_zero(&vector);
  for (int64_t i = 0; i < kHypervectorBits; ++i) {
    if (elements[i]) hdc_set_bit(&vector, static_cast<uint32_t>(i));
  }
  return vector;
}

void from_hypervector(const hdc_hypervector& vector, bool* elements) {
  for (int64_t i = 0; i < kHypervectorBits; ++i) {
    elements[i] = hdc_get_bit(&vector, static_cast<uint32_t>(i)) != 0;
  }
}

// =====================================================================
// Errors
// =====================================================================

std::string density_message(const char* label, uint32_t lanes) {
  return std::string("torchhd_sparsr: ") + label +
      " is too dense for Sparsr's CMEM: " + std::to_string(lanes) + " of " +
      std::to_string(HDC_LANES) +
      " 32-bit chunks are non-zero, but the LIL-32b compression codec used by CMEM only has "
      "room for " +
      std::to_string(HDC_MAX_STORABLE_LANES) +
      ". This is a known hardware limit, not a bug in your code: use a sparser hypervector, "
      "e.g. torchhd.random(..., vsa='BSC', sparsity=0.998). An uncompressed CMEM path that "
      "would lift this ceiling is planned but not built.";
}

// libsparsr_hdc reports what went wrong as a status code; PyTorch callers expect an
// exception. HDC_ERROR_TOO_DENSE gets an explanation of its own because it is the one a
// user hits by writing ordinary torchhd code. Its lane count is not reported here the way
// it is for an operand: the library refuses the result after computing it on the device,
// and the vector that did not fit never reaches the host.
void check_hdc(hdc_status status, const char* operation) {
  if (status == HDC_OK) return;
  TORCH_CHECK(
      status != HDC_ERROR_TOO_DENSE,
      "torchhd_sparsr: the result of ",
      operation,
      " is too dense for Sparsr's CMEM: it occupies more than the ",
      HDC_MAX_STORABLE_LANES,
      " of ",
      HDC_LANES,
      " non-zero 32-bit chunks the LIL-32b compression codec has room for. This is a known "
      "hardware limit, not a bug in your code: use sparser hypervectors. An uncompressed "
      "CMEM path that would lift this ceiling is planned but not built.");
  TORCH_CHECK(
      false,
      "torchhd_sparsr: libsparsr_hdc reported ",
      hdc_status_string(status),
      " for ",
      operation,
      ".");
}

void check_fits_device(const hdc_hypervector& vector, const char* label) {
  if (hdc_fits_device(&vector)) return;
  TORCH_CHECK(false, density_message(label, hdc_nonzero_lanes(&vector)));
}

// =====================================================================
// Storage
// =====================================================================
// A "sparsr" tensor's storage is ordinary host memory -- see the note at the top
// of this file for why it is no longer a CMEM row. The allocator still exists
// rather than reusing the CPU one because it is where the one-hypervector-per-
// tensor rule is enforced: a batched .to("sparsr") has to fail here, loudly,
// rather than being quietly split or truncated later.

void host_deleter(void* pointer) { std::free(pointer); }

struct HypervectorAllocator final : at::Allocator {
  at::DataPtr allocate(size_t nbytes) override {
    if (nbytes == 0) {
      return {nullptr, nullptr, &host_deleter, at::Device(at::DeviceType::PrivateUse1, 0)};
    }
    TORCH_CHECK(
        static_cast<int64_t>(nbytes) == kTensorBytes,
        "torchhd_sparsr: the 'sparsr' device only supports single 4096-bit "
        "hypervector tensors (",
        kTensorBytes,
        " bytes, one byte per bool element); got a request for ",
        nbytes,
        " bytes. Batched .to(\"sparsr\") tensors aren't supported yet -- "
        "move each hypervector individually.");
    void* pointer = std::calloc(1, nbytes);
    TORCH_CHECK(pointer != nullptr, "torchhd_sparsr: out of memory allocating a hypervector.");
    return {pointer, pointer, &host_deleter, at::Device(at::DeviceType::PrivateUse1, 0)};
  }

  at::DeleterFnPtr raw_deleter() const override { return &host_deleter; }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    std::memcpy(dest, src, count);
  }
};

HypervectorAllocator g_allocator;

const bool* elements_of(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_contiguous() && tensor.numel() == kHypervectorBits,
      "torchhd_sparsr: expected a contiguous 4096-element hypervector on the 'sparsr' device.");
  return static_cast<const bool*>(tensor.storage().data()) + tensor.storage_offset();
}

bool* mutable_elements_of(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_contiguous() && tensor.numel() == kHypervectorBits,
      "torchhd_sparsr: expected a contiguous 4096-element hypervector on the 'sparsr' device.");
  return static_cast<bool*>(tensor.storage().data_ptr().get()) + tensor.storage_offset();
}

// =====================================================================
// Device guard (Sparsr is a single accelerator; no multi-device indexing)
// =====================================================================

struct SparsrGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  at::DeviceType type() const override { return at::DeviceType::PrivateUse1; }
  at::Device exchangeDevice(at::Device d) const override { return at::Device(at::DeviceType::PrivateUse1, 0); }
  at::Device getDevice() const override { return at::Device(at::DeviceType::PrivateUse1, 0); }
  void setDevice(at::Device d) const override {}
  void uncheckedSetDevice(at::Device d) const noexcept override {}
  at::Stream getStream(at::Device d) const noexcept override { return at::Stream(at::Stream::DEFAULT, d); }
  at::Stream exchangeStream(at::Stream s) const noexcept override {
    return at::Stream(at::Stream::DEFAULT, at::Device(at::DeviceType::PrivateUse1, 0));
  }
  at::DeviceIndex deviceCount() const noexcept override { return 1; }
  void record(void**, const at::Stream&, const at::DeviceIndex, const c10::EventFlag) const override {
    TORCH_CHECK(false, "sparsr backend doesn't support events.");
  }
  void block(void*, const at::Stream&) const override {
    TORCH_CHECK(false, "sparsr backend doesn't support events.");
  }
  bool queryEvent(void*) const override {
    TORCH_CHECK(false, "sparsr backend doesn't support events.");
  }
  void destroyEvent(void*, const at::DeviceIndex) const noexcept override {}
  bool queryStream(const at::Stream&) const override { return true; }
  void synchronizeStream(const at::Stream&) const override {}
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, SparsrGuardImpl);

// =====================================================================
// Kernels
// =====================================================================

at::Tensor sparsr_empty_memory_format(
    at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    std::optional<at::MemoryFormat> memory_format) {
  const at::OptionalDeviceGuard device_guard(device);
  int64_t numel = 1;
  for (auto s : size) numel *= s;
  TORCH_CHECK(
      numel == kHypervectorBits,
      "torchhd_sparsr: the 'sparsr' device only supports tensors of exactly ",
      kHypervectorBits,
      " elements (one hypervector); got a shape with ",
      numel,
      " elements. Batched device-resident tensors aren't supported yet.");
  TORCH_CHECK(
      !dtype.has_value() || *dtype == at::kBool,
      "torchhd_sparsr: the 'sparsr' device only supports dtype=torch.bool (BSC hypervectors).");
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &g_allocator, private_use_ks, at::kBool, memory_format);
}

at::Tensor sparsr_empty_strided(
    at::IntArrayRef size,
    at::IntArrayRef stride,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  const at::OptionalDeviceGuard device_guard(device);
  int64_t numel = 1;
  for (auto s : size) numel *= s;
  TORCH_CHECK(
      numel == kHypervectorBits,
      "torchhd_sparsr: the 'sparsr' device only supports tensors of exactly ",
      kHypervectorBits,
      " elements (one hypervector); got a shape with ",
      numel,
      " elements.");
  TORCH_CHECK(
      !dtype.has_value() || *dtype == at::kBool,
      "torchhd_sparsr: the 'sparsr' device only supports dtype=torch.bool (BSC hypervectors).");
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(size, stride, &g_allocator, private_use_ks, at::kBool);
}

at::Tensor sparsr__copy_from(const at::Tensor& self, const at::Tensor& dst, bool /*non_blocking*/) {
  bool self_is_sparsr = self.device().type() == c10::DeviceType::PrivateUse1;
  bool dst_is_sparsr = dst.device().type() == c10::DeviceType::PrivateUse1;
  const at::OptionalDeviceGuard device_guard(self_is_sparsr ? self.device() : dst.device());

  TORCH_CHECK(
      self.numel() == kHypervectorBits && dst.numel() == kHypervectorBits,
      "torchhd_sparsr only supports hypervectors of exactly 4096 elements.");
  TORCH_CHECK(
      self.scalar_type() == at::kBool && dst.scalar_type() == at::kBool,
      "torchhd_sparsr only supports dtype=torch.bool (BSC) hypervectors.");

  // The admission check happens on the way in, not on the way out: a hypervector too dense
  // for a compressed CMEM row cannot be an operand to anything, so refusing it here is what
  // keeps every later operation from having to.
  if (!self_is_sparsr && dst_is_sparsr) {
    at::Tensor contiguous_source = self.contiguous();
    hdc_hypervector vector = to_hypervector(contiguous_source.const_data_ptr<bool>());
    check_fits_device(vector, "tensor being copied to the 'sparsr' device");
    std::memcpy(mutable_elements_of(dst), contiguous_source.const_data_ptr<bool>(), kTensorBytes);
    return dst;
  }

  if (self_is_sparsr && !dst_is_sparsr) {
    at::Tensor contiguous_dst = dst.contiguous();
    std::memcpy(contiguous_dst.data_ptr<bool>(), elements_of(self), kTensorBytes);
    if (!dst.is_contiguous()) dst.copy_(contiguous_dst);
    return dst;
  }

  if (self_is_sparsr && dst_is_sparsr) {
    std::memcpy(mutable_elements_of(dst), elements_of(self), kTensorBytes);
    return dst;
  }

  std::memcpy(dst.data_ptr(), self.data_ptr(), static_cast<size_t>(self.nbytes()));
  return dst;
}

at::Tensor sparsr__to_copy(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<at::MemoryFormat> memory_format) {
  at::Device target_device = device.value_or(self.device());
  at::ScalarType target_dtype = dtype.value_or(self.scalar_type());
  TORCH_CHECK(target_dtype == at::kBool, "torchhd_sparsr only supports dtype=torch.bool (BSC hypervectors).");

  if (target_device == self.device() && target_dtype == self.scalar_type()) {
    return self;
  }

  if (target_device.type() == c10::DeviceType::PrivateUse1) {
    at::Tensor result = sparsr_empty_memory_format({kHypervectorBits}, at::kBool, std::nullopt, target_device, std::nullopt, std::nullopt);
    sparsr__copy_from(self, result, non_blocking);
    return result;
  }

  at::Tensor result = at::empty({kHypervectorBits}, self.options().dtype(at::kBool).device(target_device));
  sparsr__copy_from(self, result, non_blocking);
  return result;
}

// Runs one two-operand HDC operation and wraps the answer back up as a tensor.
at::Tensor device_result(const at::Tensor& reference, const hdc_hypervector& value) {
  at::Tensor result = sparsr_empty_memory_format(
      {kHypervectorBits}, at::kBool, std::nullopt, reference.device(), std::nullopt, std::nullopt);
  from_hypervector(value, mutable_elements_of(result));
  return result;
}

// torchhd's bind() is `self.logical_xor(other)`, a single aten op, so registering this
// kernel is enough for it to reach Sparsr with no call-site change. libsparsr_hdc spells
// the same operation hdc_bind(), and it is one WXOR on the device either way.
at::Tensor sparsr_logical_xor(const at::Tensor& self, const at::Tensor& other) {
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  TORCH_CHECK(
      self.device().type() == c10::DeviceType::PrivateUse1 && other.device().type() == c10::DeviceType::PrivateUse1,
      "torchhd_sparsr: both operands of bind() must be on the 'sparsr' device.");

  hdc_hypervector left = to_hypervector(elements_of(self));
  hdc_hypervector right = to_hypervector(elements_of(other));
  hdc_hypervector bound;
  check_hdc(hdc_bind(&left, &right, &bound), "bind");

  return device_result(self, bound);
}

// Exposed to Python for the BSCTensor.bundle() monkeypatch (see _patches.py): bundle isn't
// a single aten op (torchhd composes it from eq/where/bernoulli_, none of which map onto
// Sparsr's bitwise instructions), so it is a direct call rather than dispatcher-level
// registration.
//
// torchhd defines the BSC bundle as `where(a == b, a, tiebreak)`, and that is exactly a
// majority vote over the three of them: where a and b agree they outvote the tiebreak two
// to one, and where they disagree the tiebreak decides. So this needs no kernel of its own
// -- hdc_bundle_majority() over three members already computes it, and the identity is
// pinned by a test in the HDC library's own harness so a change to the
// library's threshold cannot silently change torchhd's bundle.
at::Tensor bundle_kernel(const at::Tensor& a, const at::Tensor& b, const at::Tensor& tiebreak) {
  const at::OptionalDeviceGuard device_guard(at::device_of(a));
  TORCH_CHECK(
      a.device().type() == c10::DeviceType::PrivateUse1 && b.device().type() == c10::DeviceType::PrivateUse1 &&
          tiebreak.device().type() == c10::DeviceType::PrivateUse1,
      "torchhd_sparsr: bundle() operands must be on the 'sparsr' device.");

  hdc_hypervector left = to_hypervector(elements_of(a));
  hdc_hypervector right = to_hypervector(elements_of(b));
  hdc_hypervector coin = to_hypervector(elements_of(tiebreak));
  const hdc_hypervector* members[] = {&left, &right, &coin};

  hdc_hypervector bundled;
  check_hdc(hdc_bundle_majority(members, 3, &bundled), "bundle");

  return device_result(a, bundled);
}

// How much two hypervectors overlap, as the three counts libsparsr_hdc reports: the
// intersection's weight and each operand's own. Everything torchhd asks for is derived
// from those on the Python side -- the Hamming distance the BSC similarities need is
// left_weight + right_weight - 2 * overlap.
//
// The intersection and its count are one wide instruction on the device.
// Asking libsparsr_hdc for the counts rather than computing them here is what let that land
// for every caller at once, which is the point of this layering: nothing in this file
// changed when it did. The two operand weights are host counts of host-owned data, on
// purpose -- see hdc_similarity() in libsparsr_hdc.
std::tuple<int64_t, int64_t, int64_t> similarity(const at::Tensor& a, const at::Tensor& b) {
  const at::OptionalDeviceGuard device_guard(at::device_of(a));
  TORCH_CHECK(
      a.device().type() == c10::DeviceType::PrivateUse1 && b.device().type() == c10::DeviceType::PrivateUse1,
      "torchhd_sparsr: both operands of a similarity must be on the 'sparsr' device.");

  hdc_hypervector left = to_hypervector(elements_of(a));
  hdc_hypervector right = to_hypervector(elements_of(b));
  hdc_similarity_result counts;
  check_hdc(hdc_similarity(&left, &right, &counts), "similarity");

  return {static_cast<int64_t>(counts.overlap),
          static_cast<int64_t>(counts.left_weight),
          static_cast<int64_t>(counts.right_weight)};
}

// hdc_init() loads libsparsr_hdc's kernels into instruction memory and claims CMEM rows 0
// to 31. It calls sparsr_kernel_init(), which clears every memory on the device -- harmless
// here, because a "sparsr" tensor's bits live in host memory and are sent per operation.
void init_native() {
  static bool initialised = false;
  if (initialised) return;
  check_hdc(hdc_init(), "initialisation");
  initialised = true;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &sparsr_empty_memory_format);
  m.impl("empty_strided", &sparsr_empty_strided);
  m.impl("_copy_from", &sparsr__copy_from);
  m.impl("_to_copy", &sparsr__to_copy);
  m.impl("logical_xor", &sparsr_logical_xor);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_native", &init_native, "Initialise libsparsr_hdc and its device kernels.");
  m.def("bundle_kernel", &bundle_kernel, "torchhd's BSC bundle, as a majority vote over the two operands and the tiebreak.");
  m.def("similarity", &similarity, "Overlap, left weight and right weight of two 'sparsr' hypervectors.");
  // Exported so _patches.py can quote the real ceiling in its error messages
  // rather than re-typing the numbers. Two copies of one constant is how a
  // CMEM-depth mismatch (64 in the RTL, 32 in softemu) once happened.
  m.attr("LIL_MAX_NONZERO_CHUNKS") = HDC_MAX_STORABLE_LANES;
  m.attr("LIL_CHUNK_COUNT") = HDC_LANES;
}
