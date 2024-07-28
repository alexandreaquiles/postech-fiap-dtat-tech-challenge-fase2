# postech-fiap-dtat-tech-challenge-fase2

Código do Tech Challenge da Fase 2 da Pós Tech em Data Analytics, que trata de um modelo preditivo de dados da IBOVESPA por meio de forecasting de séries temporais.

## Conclusão


O objetivo do Tech Challenge foi, a partir de diferentes modelos preditivos de séries temporais, predizer o preço de fechamento da IBOVESPA.

Nos baseamos nos dados da IBOVESPA do período de 22/07/2021 a 22/07/2024.

Comparamos os seguintes modelos:
 
- ARIMA
- AutoARIMA
- ARIMA considerando retornos diários (pct_change) para trazer estacionariedade na base de dados
- Prophet considerando dados de abertura como regressor adicional
- XGBoost considerando abertura como dado exógeno
- SARIMAX também considerando abertura como dado exógeno

Considerando a data de início de `22/07/2021` e `30` dias para teste do modelo, os resultados das métricas de avaliação foram os seguintes:

| Modelo | MAE | MSE | MAPE |
|---|---|---|---|
| ARIMA | 4153.938843 | 2.523715e+07 | 3.279460 |
| AutoARIMA | 4101.737760 | 2.459460e+07 | 3.238288 |
| ARIMA (pct_change) | 3670.707297 | 1.936855e+07 | 2.903517 |
| Propjet | 721.726603 | 7.979923e+05 | 0.579813 |
| XGBoost | 800.102865 | 1.113027e+06 | 0.644785 |
| SARIMAX | 577.609337 | 5.597929e+05 | 0.465672 |

Por meio das métricas de avaliação mostradas acima, podemos concluir que o modelo com menor erro seria o **SARIMAX**.

Por isso, consideramos todos os dados disponíveis a partir de 22/07/2021 para retreinar o modelo.

Nosso objetivo final, foi considerar o próximo dia (23/07/2024) que não fazia parte da base de dados para predizer o preço de fechamento.

Como dado exógeno na nossa previsão com o modelo SARIMAX, utilizamos o dado de abertura do dia 23/07/2024: `127860`.

O modelo previu uma baixa de `0.013`% em relação ao dia anterior, resultando em `127842.97823412197`.

Entretanto, o dado histórico real de fechamento da IBOVESPA para o dia 23/07/2024 foi de `126590`. Houve uma baixa de `-0.993`% em relação ao dia anterior. A direção da previsão, de queda, foi correta porém houve uma maior magnetude na data em questão.

Entre os motivos da queda de quase 1% na IBOVESPA no dia 23/07/2024, está a desvalorização da Vale (VALE3), impulsionada por contratos futuros de minário de ferro e fraca demanda da China, segunda a revista Exame[1]. Portanto, são fatores exógenos de difícil modelagem.

[1]: https://exame.com/invest/mercados/ibovespa-hoje-23-07-2024/

## Notebook com as análises e testes

A análise exploratória dos dados, testes estatíscos, comparativos de modelos e os resultados podem ser encontrados no seguinte notebook: [tech-challenge-fase2-forecasting-ibovespa.ipynb](https://github.com/alexandreaquiles/postech-fiap-dtat-tech-challenge-fase2/blob/main/tech-challenge-fase2-forecasting-ibovespa.ipynb)

## Dashboard para comparação de modelos

Criamos um dashboard com Streamlit para tornar possível a comparação de modelos com diferentes datas de início e tamanhos do dataset de teste.

Para rodar localmente, basta ter instalado a CLI do Streamlit e executar:

```sh
streamlit run Dashboard.py
```

Caso não tenha as bibliotecas utilizadas instaladas, execute `pip install` antes de rodar o Streamlit.

Também disponibilizamos o Dashboard online no seguinte endereço: https://postech-fiap-dtat-tech-challenge-fase2-ewruwf6pn5uw5nruap4bh3.streamlit.app/

O Dashboard online apresenta certa lentidão, devido ao custo computacional de executar o treino dos modelos.

### Fonte dos dados

Dados históricos dos últimos 20 anos da IBOVESPA, considerando o período de 22/07/2004 até 22/07/2024 obtidos a partir do site Investing.com:

https://br.investing.com/indices/bovespa-historical-data