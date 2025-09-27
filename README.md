
# MVP II - Machine Learning & Analytics (40530010056_20250_01)

**Curso:** Data Science & Analytics – PUC-Rio**

**Autor:** Madson Aragão

## Visão Geral do Projeto

Este projeto tem como objetivo principal validar a viabilidade de classificar dados extraídos de células mamárias em processos de alteração celular (benignas vs. malignas) utilizando técnicas clássicas de análise de dados em Python e preparar o dataset para modelos de Machine Learning. Exploramos a relevância biológica das variáveis, avaliamos o potencial diagnóstico dos modelos e comparamos o desempenho de diferentes algoritmos de classificação.

## II. Contexto e Importância da Base de Dados

Utilizamos o **Breast Cancer Wisconsin (Diagnostic) Data Set**, disponível no UCI Machine Learning Repository. Esta base contém **569 amostras** de tumores mamários, rotuladas como **benigno (B)** ou **maligno (M)**, e **30 variáveis numéricas** que descrevem características morfológicas dos núcleos celulares, extraídas de imagens digitalizadas.

A importância desta base de dados reside no seu potencial para:

- **Diagnóstico precoce:** Auxiliar radiologistas na identificação rápida de tumores malignos.
- **Padronização de laudos:** Reduzir a subjetividade na avaliação humana.
- **Suporte à pesquisa clínica:** Correlacionar características de imagem com resultados terapêuticos.

As variáveis incluem medidas de raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal, capturadas em três escalas: média (`_mean`), erro-padrão (`_se`), e "pior" valor (`_worst`). Alterações nestas características são biologicamente relevantes para diferenciar células benignas de malignas.

## III. Exploração e Pré-processamento dos Dados

Antes de modelar, realizamos as seguintes etapas:

1.  **Carregamento e Seleção:** Importamos os dados diretamente de um repositório GitHub para reprodutibilidade. Removemos colunas irrelevantes (`id`, `Unnamed: 32`).
2.  **Análise Exploratória Visual:**
    *   Verificamos a distribuição das classes (357 benignos, 212 malignos), observando um leve desbalanceamento.
    *   Utilizamos mapas de calor para analisar a correlação entre as variáveis, identificando alta multicolinearidade entre alguns atributos (ex: `radius_mean`, `perimeter_mean`, `area_mean`).
    *   Plotagens de violino e pairplots foram usadas para visualizar a distribuição das variáveis por classe e identificar atributos com bom poder discriminatório (`area_mean`, `concave points_mean`, etc.).
3.  **Tratamento da Multicolinearidade:** Removemos variáveis altamente correlacionadas (com |r| >= 0.9) para simplificar o modelo e evitar problemas em algoritmos sensíveis a isso. Foram removidos `radius_mean`, `perimeter_mean`, `concavity_mean`, `radius_se`, `perimeter_se`, `radius_worst`, `perimeter_worst`.
4.  **Padronização:** Aplicamos `StandardScaler()` para normalizar as variáveis numéricas, garantindo que todas as features contribuam igualmente em algoritmos baseados em distância ou gradiente.
5.  **Divisão Treino/Teste:** Separamos os dados em 75% para treino e 25% para teste (`random_state=10` para reprodutibilidade), garantindo que a distribuição das classes fosse preservada nesta divisão.

## IV. Modelagem e Avaliação

Este é um problema de **classificação supervisionada binária**. Treinamos e avaliamos quatro modelos de classificação populares:

-   **K-Nearest Neighbors (KNN)**
-   **Random Forest (RF)**
-   **Support Vector Machine (SVM)**
-   **XGBoost**

Utilizamos **validação cruzada (10 folds)** no conjunto de treino para uma avaliação robusta do desempenho do modelo durante a fase de treinamento e seleção de hiperparâmetros (no caso do KNN, a validação cruzada foi usada para selecionar o melhor `k=7`). As métricas finais foram calculadas no conjunto de **teste padronizado**.

### Métricas de Avaliação

Avaliamos os modelos com as seguintes métricas:

-   **Acurácia:** Proporção de classificações corretas.
-   **MCC (Matthews Correlation Coefficient):** Medida robusta de correlação entre a predição e o valor real, útil em classes desbalanceadas.
-   **AUC-ROC:** Área sob a curva Receiver Operating Characteristic, indica a capacidade do modelo de discriminar entre as classes.
-   **Matriz de Confusão:** Apresenta True Positives (VP), True Negatives (VN), False Positives (FP) e False Negatives (FN). Em diagnóstico médico, minimizar FN (classificar maligno como benigno) é crucial.
-   **Precisão (Precision):** Proporção de identificações positivas que foram realmente corretas.
-   **Recall (Sensibilidade):** Proporção de casos positivos reais que foram corretamente identificados.

### Resultados Comparativos no Conjunto de Teste

| Modelo        | Acurácia | MCC    | AUC-ROC | VP | VN | FP | FN | Precisão (Benigno) | Recall (Benigno) | Precisão (Maligno) | Recall (Maligno) |
|---------------|----------|--------|---------|----|----|----|----|--------------------|------------------|--------------------|------------------|
| KNN           | 0.9580   | 0.9090 | 0.9968  | 48 | 89 | 2  | 4  | 0.9570             | 0.9780           | 0.9600             | 0.9231           |
| Random Forest | 0.9860   | 0.9705 | 0.9987  | 52 | 89 | 2  | 0  | 1.0000             | 0.9780           | 0.9630             | 1.0000           |
| SVM           | 0.9860   | 0.9705 | 1.0000  | 52 | 89 | 2  | 0  | 1.0000             | 0.9780           | 0.9630             | 1.0000           |
| XGBoost       | 0.9720   | 0.9396 | 0.9985  | 50 | 89 | 2  | 2  | 0.9780             | 0.9780           | 0.9615             | 0.9615           |

*As métricas VP, VN, FP, FN referem-se à classe 'Maligno'.*

## V. Principais Achados e Conclusões

A análise comparativa revela que todos os modelos apresentaram alto desempenho na classificação de tumores mamários, com acurácia superior a 95% e AUC-ROC acima de 0.99.

-   **Modelos de Destaque:** **Random Forest** e **SVM** se destacaram por atingirem a maior acurácia e MCC, e, fundamentalmente, por apresentarem **zero Falsos Negativos (FN = 0)** no conjunto de teste. Isso significa que eles foram capazes de identificar corretamente todos os casos malignos no teste, o que é um resultado crítico em aplicações médicas. O SVM ainda obteve uma AUC-ROC perfeita (1.0000).
-   **Desempenho Forte:** **XGBoost** também demonstrou um desempenho muito robusto, com métricas elevadas e um número baixo de Falsos Negativos (FN = 2). É uma excelente alternativa, especialmente em cenários onde a escalabilidade e a eficiência computacional são importantes.
-   **Modelo Base Sólido:** O **KNN**, apesar de ser o modelo com menor desempenho entre os avaliados (FN = 4), ainda assim obteve resultados muito bons, validando a qualidade das features e a abordagem de pré-processamento.

A cuidadosa etapa de pré-processamento, incluindo a análise de correlação e a padronização, foi essencial para o sucesso de todos os modelos.

## VI. Como Usar o Notebook

Para replicar a análise e os resultados deste projeto, siga os passos abaixo:

1.  **Abrir no Google Colab:** Clique no botão "Open in Colab" no topo deste README (se disponível) ou faça upload do arquivo `.ipynb` para o Google Colab.
2.  **Executar as Células:** Execute as células do notebook sequencialmente, de cima para baixo. Certifique-se de que cada célula seja concluída antes de passar para a próxima.
3.  **Acesso aos Dados:** O notebook carrega os dados diretamente do repositório GitHub, então não é necessário fazer download manual do arquivo CSV.
4.  **Modelos Pré-Treinados:** Os modelos treinados (KNN, Random Forest, SVM, XGBoost) são salvos no diretório `modelos_pre_treinados` após a execução da seção correspondente. Você também pode baixá-los diretamente do repositório GitHub para uso posterior sem a necessidade de retreinamento.

## VII. Modelos Pré-Treinados

Os modelos finais treinados neste notebook foram salvos e estão disponíveis para download direto no seguinte diretório do repositório GitHub:

[https://github.com/madsondeluna/mvp_pucrio_data_analysis/tree/main/modelos_pre_treinados](https://github.com/madsondeluna/mvp_pucrio_data_analysis/tree/main/modelos_pre_treinados)

Você pode carregar esses modelos em outro ambiente Python usando a biblioteca `pickle`.

## Próximos Passos Sugeridos

Com base nesta análise inicial, futuras explorações podem incluir:

-   Otimização mais aprofundada de hiperparâmetros para os modelos Random Forest, SVM e XGBoost.
-   Análise de importância das features para entender quais características morfológicas são mais preditivas para cada modelo.
-   Testes com conjuntos de dados externos para validar a generalização dos modelos.
-   Investigação dos casos de Falsos Positivos para identificar padrões e possíveis melhorias no pré-processamento ou modelagem.
-   Exploração de modelos de Deep Learning para análise direta das imagens.
