// Backward-compat shim for PyTorch 2.5 -> 2.6 SetDevice signature change.
// Some prebuilt artifacts still reference c10::cuda::SetDevice(DeviceIndex, bool).
// PyTorch 2.6+ exposes c10::cuda::SetDevice(DeviceIndex) only, so we provide
// the old mangled symbol and forward to the current API.
#include <cstdint>

#ifndef C10_API
#define C10_API __attribute__((visibility("default")))
#endif

namespace c10 {
using DeviceIndex = int16_t;

namespace cuda {
// Forward declaration of the new signature shipped in torch 2.6+
C10_API int SetDevice(DeviceIndex device);

// Old mangled symbol expected by existing binaries.
C10_API __attribute__((used)) int SetDevice(DeviceIndex device, bool /*unused*/) {
    return SetDevice(device);
}
}  // namespace cuda

// Force the linker to keep the old symbol even if it appears unreferenced.
namespace {
using SetDeviceBoolSig = int (*)(c10::DeviceIndex, bool);
[[gnu::used]] static SetDeviceBoolSig _force_link_setdevice_bool =
    (SetDeviceBoolSig)&c10::cuda::SetDevice;
}

extern "C" void ktransformers_force_link_setdevice() {
    (void)_force_link_setdevice_bool;
}

// Explicitly export the legacy mangled symbol expected by older binaries.
extern "C" __attribute__((visibility("default"))) int
_ZN3c104cuda9SetDeviceEab(short device, bool unused) {
    return c10::cuda::SetDevice(device, unused);
}

extern "C" __attribute__((visibility("default"))) int
_ZN3c104cuda9SetDeviceEs(short device) {
    return c10::cuda::SetDevice(device);
}
}  // namespace c10

