# MLX Swift – "Constexpr if is a C++17 extension" warnings

When building with the **mlx-swift** package (e.g. in Cadenza-Audio), you may see Metal compiler warnings like:

```
.../mlx-swift/Source/Cmlx/mlx-generated/metal/steel/attn/kernels/steel_attention.h:359:16 Constexpr if is a C++17 extension
.../steel_attention.h:429:14 Constexpr if is a C++17 extension
.../steel_attention.h:439:14 Constexpr if is a C++17 extension
```

## Cause

The Metal steel attention kernels (`.metal` files that `#include` C++ headers) use `if constexpr`, which is C++17. The package sets `cxxLanguageStandard: .gnucxx17` at the **package** level, but the **Metal** compiler that builds those `.metal` files may not receive that setting and can default to an earlier C++ standard.

## Fix (if warnings reappear)

If the warnings come back (e.g. after resolving packages or a clean build), the fix is to pass C++17 explicitly to the Cmlx target in **mlx-swift**’s `Package.swift`:

1. Use a **local package override** for mlx-swift (e.g. clone `https://github.com/ml-explore/mlx-swift` and add it as a local path dependency in your app’s project or in AceStepSwift’s `Package.swift`), **or**
2. Rely on an upstream change in mlx-swift.

In the **Cmlx** target, add to `cxxSettings`:

```swift
// Metal steel kernels use C++17 (e.g. if constexpr in steel_attention.h)
.unsafeFlags(["-std=c++17"]),
```

So the `cxxSettings` block starts like this:

```swift
cxxSettings: [
    .headerSearchPath("mlx"),
    .headerSearchPath("mlx-c"),
    .headerSearchPath("metal-cpp"),
    .headerSearchPath("json/single_include/nlohmann"),
    .headerSearchPath("fmt/include"),

    // Metal steel kernels use C++17 (e.g. if constexpr in steel_attention.h)
    .unsafeFlags(["-std=c++17"]),

    .define("MLX_USE_ACCELERATE"),
    // ... rest unchanged
],
```

## One-off fix in DerivedData (temporary)

You can edit the **resolved** package in DerivedData so the next build uses C++17. The edit will be lost the next time Xcode/SPM re-resolves or re-checks out the package.

Path (replace with your app name and DerivedData id):

```
~/Library/Developer/Xcode/DerivedData/<YourApp>-<hash>/SourcePackages/checkouts/mlx-swift/Package.swift
```

Add the `.unsafeFlags(["-std=c++17"])` line to the Cmlx target’s `cxxSettings` as above.

## Upstream

Consider opening an issue or PR on [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) so the Cmlx target explicitly passes C++17 to the Metal/C++ compiler, making the package build cleanly without warnings.
