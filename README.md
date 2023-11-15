# BDF

<p align="center">
<a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202-blue" alt="Licensing"/></a>
<!-- <a href="https://gitlab.com/JonasHarsch/cardillo3/-/tree/main"><img src="https://gitlab.com/JonasHarsch/cardillo3/badges/main/pipeline.svg" alt="gitlab pipeline status"/></a> -->
<a href="https://www.python.org/dev/peps/pep-0008/"><img alt="Code standard: PEP8" src="https://img.shields.io/badge/code%20standard-PEP8-black"></a>
</p>

BDF solver for implicit differential algebraic equations of the form

```
0 = F(t, x, x')
x(t0) = x0
x'(t0) = x'0
```