# IADT-Fase1-Tech-challengeB

## Guia de avaliação do projeto

Este repositório foi organizado para facilitar a análise do projeto por professor, banca ou qualquer pessoa responsável pela avaliação.

O trabalho apresenta uma solução de **apoio ao diagnóstico de câncer de mama com Machine Learning**, utilizando o **Breast Cancer Wisconsin Diagnostic Dataset**. O foco principal do projeto é comparar modelos de classificação e priorizar a identificação correta de casos malignos, com atenção especial à métrica de **recall**.

## Leitura rápida para avaliação

Se a ideia for fazer uma avaliação rápida, a sequência recomendada é:

1. Ler o relatório final em `reports/relatorio_final_fiap_completo.pdf`.
2. Se preferir navegar em formato web, abrir `reports/relatorio_final_fiap_completo.html`.
3. Caso queira reproduzir a execução, subir o ambiente com Docker.
4. Abrir os notebooks em ordem, principalmente `06_data_division.ipynb` e `07_model_comparison_and_tuning.ipynb`, onde está a modelagem e a análise final.

## Estrutura do projeto

```text
IADT-Fase1-Tech-challengeB/
├── data/                          # Dataset original e base tratada
├── docs/                          # Materiais de apoio e documentação do desafio
├── notebooks/                     # Pipeline principal do projeto
├── reports/
│   ├── figures/                   # Imagens usadas no relatório
│   ├── tables/                    # Tabelas em CSV geradas pelos notebooks
│   ├── relatorio_final_fiap_completo.md
│   ├── relatorio_final_fiap_completo.html
│   └── relatorio_final_fiap_completo.pdf
├── src/                           # Pasta reservada para scripts futuros
├── Dockerfile                     # Ambiente com Python + Jupyter Notebook
├── requirements.txt               # Dependências Python
└── README.md                      # Este guia de avaliação
```

## Como executar com Docker

O projeto já possui um `Dockerfile` configurado para subir um ambiente com Python 3.11 e Jupyter Notebook.

### Pré-requisito

- Ter o Docker instalado e em funcionamento.

### 1. Construir a imagem

```bash
docker build -t iadt-fase1-tech-chalenge-b .
```

### 2. Executar o container

```bash
docker run --rm -p 8888:8888 --name iadt-fase1-tech-chalenge-b iadt-fase1-tech-chalenge-b
```

### 3. Acessar o ambiente

Após subir o container, abrir no navegador:

```text
http://localhost:8888/tree
```

O ambiente inicia o **Jupyter Notebook** já apontando para a raiz do projeto.

### Observações úteis

- Se o nome do container já estiver em uso, remova o container anterior antes de executar novamente:

```bash
docker rm -f iadt-fase1-tech-chalenge-b
```

- Se a porta `8888` estiver ocupada, troque o mapeamento, por exemplo:

```bash
docker run --rm -p 8889:8888 --name iadt-fase1-tech-chalenge-b iadt-fase1-tech-chalenge-b
```

Nesse caso, o acesso passa a ser por `http://localhost:8889/tree`.

## Como usar os notebooks

Os notebooks foram organizados para contar a evolução do projeto do início ao fim. A recomendação é executá-los na ordem numérica.

### Ordem sugerida

- `notebooks/01_exploracao_dados.ipynb`  
  Faz a análise exploratória inicial da base, verificando estrutura, tipos, estatísticas, valores ausentes e distribuição da variável alvo.

- `notebooks/02_clean.ipynb`  
  Remove colunas desnecessárias, como identificadores sem valor preditivo.

- `notebooks/03_correlation.ipynb`  
  Analisa correlação entre atributos e a variável alvo.

- `notebooks/04_graphics.ipynb`  
  Gera visualizações para entender a distribuição das classes.

- `notebooks/05_data_division.ipynb`  
  Inicia a modelagem: separa `X` e `y`, divide treino e teste, aplica `StandardScaler` e executa a primeira avaliação dos modelos.

- `notebooks/06_model_comparison_and_tuning.ipynb`  
  Consolida a parte principal da análise: comparação de modelos, matrizes de confusão, validação cruzada, `GridSearchCV`, interpretabilidade, ajuste de threshold e geração dos artefatos finais usados no relatório.

### Dica para avaliação

Se o objetivo for entender rapidamente o valor analítico do projeto, os dois notebooks mais importantes são:

- `05_data_division.ipynb`
- `06_model_comparison_and_tuning.ipynb`

Os notebooks anteriores mostram a preparação da base e justificam as transformações aplicadas.

## Como visualizar o relatório final

O relatório final já está disponível em três formatos dentro da pasta `reports/`.

### PDF

Arquivo principal para leitura final:

```text
reports/relatorio_final_fiap_completo.pdf
```

### HTML

Versão navegável em navegador:

```text
reports/relatorio_final_fiap_completo.html
```

### Markdown

Versão-fonte editável do relatório:

```text
reports/relatorio_final_fiap_completo.md
```

### Imagens e tabelas do relatório

- As imagens utilizadas no relatório estão em `reports/figures/`.
- As tabelas exportadas em CSV estão em `reports/tables/`.

## Tecnologias principais

O projeto utiliza principalmente:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`
- `shap`

## Observações finais para avaliação

- O repositório já contém os artefatos finais do trabalho, então não é obrigatório reexecutar tudo para visualizar os resultados.
- A execução via Docker é a forma mais simples de reprodução do ambiente.
- Os notebooks usam caminhos relativos dentro da própria estrutura do projeto, o que facilita a reprodução da análise.
- O relatório final pode ser lido diretamente sem depender da execução prévia dos notebooks.
