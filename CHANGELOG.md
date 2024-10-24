# Changelog

## [0.151.0](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.150.9...v0.151.0) (2024-10-18)


### ⚠ BREAKING CHANGES

* fflonk ([#28](https://github.com/matter-labs/zksync-crypto-gpu/issues/28))

### Features

* Bump `zksync-protocol` to 0.150.6 ([#34](https://github.com/matter-labs/zksync-crypto-gpu/issues/34)) ([43704f3](https://github.com/matter-labs/zksync-crypto-gpu/commit/43704f3e3caa25bbe11780b0530c65e93c035c8e))
* enable compilation without Bellman CUDA ([#31](https://github.com/matter-labs/zksync-crypto-gpu/issues/31)) ([39860f5](https://github.com/matter-labs/zksync-crypto-gpu/commit/39860f574def8fdb547099afb341019afe8bdf47))
* fflonk ([#28](https://github.com/matter-labs/zksync-crypto-gpu/issues/28)) ([acd71d8](https://github.com/matter-labs/zksync-crypto-gpu/commit/acd71d80584fa6099180ed4257811783e5dc46f1))


### Bug Fixes

* Attempt to pin github runner to 22.04.1 ([#36](https://github.com/matter-labs/zksync-crypto-gpu/issues/36)) ([ddb7311](https://github.com/matter-labs/zksync-crypto-gpu/commit/ddb7311af01aff8430a7657c24eb1d2576a8c51d))


### Reverts

* "feat!: fflonk" ([#33](https://github.com/matter-labs/zksync-crypto-gpu/issues/33)) ([dccaaa6](https://github.com/matter-labs/zksync-crypto-gpu/commit/dccaaa6103950b242c3bfc548ca77a3cb1d2af37))

## [0.150.9](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.150.8...v0.150.9) (2024-09-24)


### Features

* enable compilation with CUDA stubs via the `ZKSYNC_USE_CUDA_STUBS` environment variable ([#29](https://github.com/matter-labs/zksync-crypto-gpu/issues/29)) ([f77ff80](https://github.com/matter-labs/zksync-crypto-gpu/commit/f77ff80ee4bbe6d83ffab9b9915fc922ef87c1ad))

## [0.150.8](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.150.7...v0.150.8) (2024-09-10)


### Features

* **gpu-ffi:** add bindings for the distribute function in bellman-cuda ([#22](https://github.com/matter-labs/zksync-crypto-gpu/issues/22)) ([a099924](https://github.com/matter-labs/zksync-crypto-gpu/commit/a099924566592ea0587c27ff26ea2e0f742f775f))


### Bug Fixes

* **boojum-cuda:** "un-swap" coarse and fine count for powers_data_g_i ([#27](https://github.com/matter-labs/zksync-crypto-gpu/issues/27)) ([95b29da](https://github.com/matter-labs/zksync-crypto-gpu/commit/95b29da7581121b1e2ffc068d90725a9616f7123))

## [0.150.7](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.150.6...v0.150.7) (2024-09-06)


### Features

* Bump protocol and crypto deps ([#23](https://github.com/matter-labs/zksync-crypto-gpu/issues/23)) ([e15fdd2](https://github.com/matter-labs/zksync-crypto-gpu/commit/e15fdd2720827e45cfb895debc078c6514b9cbaa))

## [0.150.6](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.150.5...v0.150.6) (2024-09-04)


### Bug Fixes

* **shivini:** pub use ProverContextConfig ([#20](https://github.com/matter-labs/zksync-crypto-gpu/issues/20)) ([fb3c8e7](https://github.com/matter-labs/zksync-crypto-gpu/commit/fb3c8e7f36998fe3ab37a67dc808d7b4624e87a4))

## [0.150.5](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.150.4...v0.150.5) (2024-09-04)


### Features

* **ci:** Introduce CI for automatic releases ([#11](https://github.com/matter-labs/zksync-crypto-gpu/issues/11)) ([847059a](https://github.com/matter-labs/zksync-crypto-gpu/commit/847059a4222b44de109eec4856d2ea70fd9ab8d3))
* implement ProverContextConfig ([#10](https://github.com/matter-labs/zksync-crypto-gpu/issues/10)) ([0c6ba4b](https://github.com/matter-labs/zksync-crypto-gpu/commit/0c6ba4bd2bf759f4d3b594b26854a10ef69d5c68))
