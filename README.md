# mathbr

`mathbr` é uma biblioteca educacional de modelos estatísticos e de machine learning escritos em C++17 e expostos para Python com pybind11. O projeto implementa os cálculos diretamente, para facilitar o estudo de C++, pybind11 e dos métodos matemáticos.

**Versão 0.6.0 — em desenvolvimento.** A API pode mudar e a biblioteca ainda não é indicada para produção.

## O que tem

- Ativações e perdas: sigmoid, ReLU, GELU, softmax, MSE, MAE, RMSE e log loss.
- Regressão: linear e logística por gradiente, OLS, WLS, Ridge, Lasso e Elastic Net.
- Econometria: IV/2SLS e efeitos fixos por entidade.
- Séries temporais: AR(p), VAR(p), ARCH(1) e GARCH(1,1).

## Instalação

Requer Python 3.9+ e um compilador C++17. No Windows, instale o Microsoft C++ Build Tools.

```bash
python -m pip install .
```

## Exemplo rápido

```python
import mathbr

model = mathbr.OLS(n_features=1)
model.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 8.0])

print(model.coefficients())  # intercepto e inclinação: aproximadamente [0.8, 2.3]
print(model.r_squared())
print(model.predict([4.0]))
```

Para entender **todos os modelos, exemplos, hipóteses e limitações**, leia [PROJECT.md](PROJECT.md). A organização do código e as decisões técnicas estão em [ARCHITECTURE.md](ARCHITECTURE.md).
