# Guia do projeto mathbr

`mathbr` é uma extensão Python compilada em C++17. Ela reúne funções matemáticas, regressões, métodos econométricos e modelos de séries temporais. O objetivo principal é estudar as implementações; os resultados não substituem uma biblioteca estatística de produção. A versão atual é 0.6.0.

## Instalação e testes

O projeto requer Python 3.9+ e um compilador C++17. O build via `pip` instala `pybind11` automaticamente. No Windows, instale o Microsoft C++ Build Tools antes de compilar.

```bash
python -m pip install .
```

Para desenvolver:

```bash
python -m pip install -e . pytest
python -m pytest
```

O GitHub Actions compila e testa em Linux e Windows a cada push ou pull request. `pyproject.toml` contém os metadados e dependências de build; `setup.py` lista os arquivos C++ da extensão.

## API numérica

`mathbr.activations` oferece `sigmoid`, `tanh_activation`, `relu`, `leaky_relu`, `gelu`, `swish` e `softmax`, além das derivadas escalares das ativações, exceto a derivada de softmax. `softmax` recebe um vetor não vazio e devolve probabilidades que somam 1.

`mathbr.losses` oferece `mse`, `mse_derivative`, `mae`, `rmse` e `logloss`. As funções de perda exigem entradas do mesmo tamanho e não vazias. `logloss` recebe rótulos 0/1 e probabilidades no intervalo [0, 1]. Entradas inválidas lançam `ValueError` no Python.

```python
import mathbr

print(mathbr.activations.softmax([1.0, 2.0, 3.0]))
print(mathbr.losses.mse([1.0, 2.0], [1.1, 1.9]))
```

## Regressão e classificação

| Classe | Ajuste | Resultados principais |
| --- | --- | --- |
| `LinearRegression(n_features)` | Gradiente descendente full-batch para MSE | `get_weights()`, `get_bias()`, `predict()`, `predict_batch()` |
| `LogisticRegression(n_features)` | Gradiente descendente full-batch para log loss binária | `get_weights()`, `get_bias()`, `predict()`, `predict_proba()`, `predict_proba_batch()` |
| `OLS(n_features)` | Mínimos quadrados por QR | `coefficients()`, `standard_errors()`, `r_squared()`, `residual_variance()`, `degrees_of_freedom()` |
| `WLS(n_features)` | Mínimos quadrados ponderados por QR | Mesmos resultados do OLS, com pesos por observação |
| `Ridge`, `Lasso`, `ElasticNet` | Descida cíclica por coordenadas | `coefficients()`, `intercept()`, `predict()`, `iterations()`, `converged()` |

`LinearRegression.fit(X, y, lr=0.01, epochs=1000)` e `LogisticRegression.fit(X, y, lr=0.01, epochs=1000)` recebem uma lista de linhas e alvos. Para a regressão logística, os alvos precisam ser 0 ou 1. `predict` usa limiar 0,5 por padrão e aceita outro limiar entre 0 e 1.

`OLS.fit(X, y)` inclui intercepto automaticamente. Seu vetor de coeficientes começa pelo intercepto, seguido pelas inclinações. Exige mais observações do que parâmetros, valores finitos e uma matriz de desenho de posto completo. Os erros-padrão são os clássicos, sob independência e variância constante dos erros.

`WLS.fit(X, y, weights)` tem a mesma ordem de coeficientes e exige pesos estritamente positivos e finitos. Os erros-padrão clássicos de WLS pressupõem pesos proporcionais ao inverso das variâncias dos erros.

```python
ols = mathbr.OLS(n_features=1)
ols.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 8.0])
print(ols.coefficients(), ols.standard_errors(), ols.r_squared())

wls = mathbr.WLS(n_features=1)
wls.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 4.0], [1.0, 1.0, 0.1])
print(wls.coefficients())
```

`Ridge`, `Lasso` e `ElasticNet` usam `fit(X, y)`. Seus construtores recebem `n_features`, `alpha=1.0`, `max_iter=1000` e `tol=1e-8`. Elastic Net também recebe `l1_ratio=0.5`. A função objetivo é `SSE/(2n) + alpha * [l1_ratio * ||coef||_1 + (1-l1_ratio) * ||coef||_2²/2]`. Ridge fixa `l1_ratio=0`; Lasso fixa `l1_ratio=1`. O intercepto não é penalizado. As variáveis **não são padronizadas automaticamente**, então sua escala afeta a penalização. Verifique `converged()` após o ajuste.

```python
ridge = mathbr.Ridge(n_features=1, alpha=1.0)
ridge.fit([[-1.0], [0.0], [1.0]], [-2.0, 0.0, 2.0])
print(ridge.intercept(), ridge.coefficients(), ridge.converged())
```

## Econometria

`IV2SLS(n_exog, n_endog, n_instruments)` usa regressão em dois estágios. `fit(exog, endog, instruments, y)` recebe, em matrizes separadas, as variáveis exógenas, endógenas e os instrumentos excluídos. O resultado de `coefficients()` segue a ordem: intercepto, coeficientes exógenos, coeficientes endógenos. `first_stage_r_squared()` apresenta o R² geral de cada primeiro estágio; **não é um teste de instrumentos fracos**. Os instrumentos precisam ser relevantes e exógenos para uma interpretação causal, o que a classe não consegue verificar. A classe não expõe erros-padrão, pois os erros-padrão de OLS do segundo estágio seriam incorretos para 2SLS.

```python
iv = mathbr.IV2SLS(n_exog=0, n_endog=1, n_instruments=1)
iv.fit(exog=[[], [], [], [], []],
       endog=[[0], [2], [1], [4], [3]],
       instruments=[[0], [1], [2], [3], [4]],
       y=[1, 5, 3, 9, 7])
print(iv.coefficients())
```

`FixedEffects(n_features)` estima inclinações usando a variação **dentro** de cada entidade. `fit(X, y, entity_ids)` aceita IDs inteiros; `coefficients()` retorna as inclinações, `entity_intercept(id)` retorna o intercepto estimado da entidade e `within_r_squared()` mede o ajuste após retirar as médias por entidade. `predict(x, entity_id)` funciona apenas para entidades presentes no ajuste. Variáveis constantes dentro de cada entidade não são identificáveis. A implementação não inclui efeitos fixos de tempo nem erros-padrão agrupados.

```python
fe = mathbr.FixedEffects(n_features=1)
fe.fit([[0], [1], [2], [0], [1], [2]],
       [1, 3, 5, 10, 12, 14],
       [10, 10, 10, 20, 20, 20])
print(fe.coefficients(), fe.entity_intercept(20), fe.within_r_squared())
```

## Séries temporais e volatilidade

`AR(lags)` ajusta uma série por regressão nos próprios valores anteriores. `VAR(n_series, lags)` faz o mesmo com várias séries e devolve uma equação por variável. Ambos recebem observações em ordem temporal com `fit(...)` e fazem previsões recursivas por `forecast(steps)`. Em VAR, cada linha de `coefficients()` contém intercepto, todas as séries na defasagem 1, depois na defasagem 2 e assim por diante. `residual_covariance()` devolve a matriz de covariância residual.

```python
ar = mathbr.AR(lags=1)
ar.fit([1, 2, 4, 8, 16, 32])
print(ar.coefficients(), ar.forecast(2))
```

`ARCH(max_iter=2000, tol=1e-7)` implementa ARCH(1). `GARCH(max_iter=2000, tol=1e-7)` implementa GARCH(1,1). Ambos ajustam média constante e variância condicional por quase-máxima verossimilhança gaussiana. Em GARCH, `h[t] = omega + alpha * erro[t-1]² + beta * h[t-1]`; em ARCH, `beta=0`. A API expõe `mean()`, `omega()`, `alpha()`, `beta()`, `log_likelihood()`, `conditional_variance()`, `forecast_variance(steps)` e `converged()`. O ajuste exige ao menos 20 observações com variância positiva. O otimizador é uma busca simples por coordenadas; confira `converged()` antes de interpretar os parâmetros.

```python
garch = mathbr.GARCH()
garch.fit([-2.0, -1.0, 0.5, 1.0, 2.0] * 20)
print(garch.omega(), garch.alpha(), garch.beta())
print(garch.forecast_variance(3), garch.converged())
```

AR e VAR não incluem seleção automática de defasagens, testes de estacionariedade ou intervalos de previsão. ARCH e GARCH não incluem erros-padrão, assimetria ou distribuições de inovações além da gaussiana.

## Organização e limites gerais

`main.cpp` define o módulo pybind11 e contém as funções de ativação, perdas e as duas regressões por gradiente. Os modelos adicionais têm declarações em `include/mathbr/` e implementações em `src/`. Os testes estão em `tests/`. A descrição das dependências entre componentes e das decisões técnicas está em [ARCHITECTURE.md](ARCHITECTURE.md).

As interfaces aceitam listas Python convertidas para `std::vector`, o que pode copiar dados. Ainda não há integração direta com buffers NumPy, fórmulas ou DataFrames. Este é um projeto educacional em evolução; confira as hipóteses de cada modelo antes de usar os resultados em uma análise.
