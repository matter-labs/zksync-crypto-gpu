# Changelog

## [0.152.6](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.152.5...v0.152.6) (2024-11-19)


### Miscellaneous Chores

* bump crypto crates to 0.30.8 and protocol crates to 0.150.14 ([#55](https://github.com/matter-labs/zksync-crypto-gpu/issues/55)) ([9eff2ac](https://github.com/matter-labs/zksync-crypto-gpu/commit/9eff2ac658376e85c2fa333b168b699885b9cda5))

## [0.152.5](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.152.4...v0.152.5) (2024-11-18)


### Miscellaneous Chores

* bump crypto and protocol crates ([#52](https://github.com/matter-labs/zksync-crypto-gpu/issues/52)) ([b05515a](https://github.com/matter-labs/zksync-crypto-gpu/commit/b05515af18e38961051a1bf6e40de9d45bd2049c))

## [0.152.4](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.152.3...v0.152.4) (2024-11-06)


### Features

* Bump protocol versions ([#49](https://github.com/matter-labs/zksync-crypto-gpu/issues/49)) ([cb24eb6](https://github.com/matter-labs/zksync-crypto-gpu/commit/cb24eb60958d0950d5ff7c772562d92930fa2577))

## [0.152.3](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.152.2...v0.152.3) (2024-10-31)


### Bug Fixes

* **proof-compression:** add missing crate description ([#46](https://github.com/matter-labs/zksync-crypto-gpu/issues/46)) ([ba077db](https://github.com/matter-labs/zksync-crypto-gpu/commit/ba077dbd497310c14bfb574888813e372d0b693a))

## [0.152.2](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.152.1...v0.152.2) (2024-10-31)


### Features

* **proof-compression:** release ([#44](https://github.com/matter-labs/zksync-crypto-gpu/issues/44)) ([00ba9a1](https://github.com/matter-labs/zksync-crypto-gpu/commit/00ba9a1ede62e71aa20a3e37870725adad89ba0b))

## [0.152.1](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.152.0...v0.152.1) (2024-10-31)


### Features

* fflonk gpu implementation ([#26](https://github.com/matter-labs/zksync-crypto-gpu/issues/26)) ([9d11084](https://github.com/matter-labs/zksync-crypto-gpu/commit/9d11084cec1bd1b88de9a28524923a5217ebd0ad))

## [0.152.0](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.151.1...v0.152.0) (2024-10-31)


### ⚠ BREAKING CHANGES

* fflonk ([#38](https://github.com/matter-labs/zksync-crypto-gpu/issues/38))

### Features

* fflonk ([#38](https://github.com/matter-labs/zksync-crypto-gpu/issues/38)) ([33ed62a](https://github.com/matter-labs/zksync-crypto-gpu/commit/33ed62afab7bb3606f7e428c354cfee4af03de49))

## [0.151.1](https://github.com/matter-labs/zksync-crypto-gpu/compare/v0.151.0...v0.151.1) (2024-10-25)


### Bug Fixes

* bump `zksync-protocol` to 0.150.7 ([#39](https://github.com/matter-labs/zksync-crypto-gpu/issues/39)) ([9df8ec0](https://github.com/matter-labs/zksync-crypto-gpu/commit/9df8ec04001e6d09a0465aaf957895aefefe122d))

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
