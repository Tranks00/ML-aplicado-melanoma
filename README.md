# Reprodução Científica: Classificação de Imagens de Câncer de Pele com VGG19 e Machine Learning Tradicional

Este repositório contém o código e os resultados do projeto de reprodução científica para a disciplina de Aprendizagem de Máquina, baseado no artigo:

**Título Original:** "Aplicação de Redes de Aprendizado Profundo e Algoritmos de Aprendizado de Máquina para Classificar Imagens de Câncer de Pele" [1]

---

## Resumo do Projeto

O objetivo deste projeto é analisar e reproduzir a metodologia proposta no artigo citado. O método combina a extração de características de imagens de lesões de pele, utilizando a rede convolucional VGG19 pré-treinada [3], com a classificação subsequente por meio de algoritmos tradicionais de Machine Learning, como SVM, Regressão Logística e Perceptron [1, 4].

O projeto foi implementado em um notebook do Google Colab e utiliza o dataset público **HAM10000** [2]. Para otimizar o tempo de execução dos experimentos, foi utilizada uma amostra estratificada correspondente a 30% do conjunto de dados original.

Os resultados obtidos validaram as conclusões do artigo, demonstrando o desafio imposto pelo desbalanceamento de classes, que resulta em alta acurácia, mas baixo F1-Score.

---

## Como Configurar o Ambiente e Executar os Experimentos
...

Este projeto foi projetado para ser executado no ambiente do Google Colab, que já vem com a maioria das dependências pré-instaladas. Siga os passos abaixo para replicar os resultados.

### Pré-requisitos

1.  **Conta Google:** Para usar o Google Colab.
2.  **Conta Kaggle:** O dataset é baixado diretamente do Kaggle. Você precisará de um token de API.

### Passo a Passo

**1. Obtenha seu Token da API do Kaggle**

   - Faça login na sua conta do **Kaggle**.
   - Vá para a página da sua conta (`https://www.kaggle.com/account`).
   - Na seção "API", clique no botão **"Create New API Token"**.
   - Um arquivo chamado `kaggle.json` será baixado para o seu computador. Guarde este arquivo.

**2. Abra o Notebook no Google Colab**

   - Faça o upload do arquivo `.ipynb` deste repositório para o seu Google Drive.
   - Abra o notebook com o Google Colab.

**3. Execute o Script Principal**
 - O notebook foi estruturado para instalar a dependência kaleido que já está no script principal.
      ```python
     !pip install kaleido
     ```
   - Execute a célula principal que contém todo o pipeline do projeto.
   - A primeira etapa do script (`ETAPA 0`) irá pedir para você fazer o upload de um arquivo.
   - Clique no botão **"Escolher arquivos"** e selecione o `kaggle.json` que você baixou no Passo 1.

**4. Aguarde a Execução**

   - Após o upload do `kaggle.json`, o script continuará a execução automaticamente:
     - **Download e Descompactação:** Baixará o dataset do Kaggle (aproximadamente 5.2 GB).
     - **Extração de Atributos:** Processará as imagens da amostra com a VGG19. Esta é a etapa mais demorada.
     - **Treinamento e Avaliação:** Treinará os 7 classificadores.
     - **Geração de Resultados:** Exibirá a tabela de métricas e o gráfico de radar final.

O processo completo, utilizando a amostra de 30% dos dados, deve levar entre 58 a 60 minutos, dependendo dos recursos alocados pelo Colab.

---

## Referências

[1] SÁ, João P. C. A. de; ENSINA, Leandro A.; JERONYMO, Daniel C.. Aplicação de Redes de Aprendizado Profundo e Algoritmos de Aprendizado de Máquina para Classificar Imagens de Câncer de Pele. In: **SIMPÓSIO BRASILEIRO DE COMPUTAÇÃO APLICADA À SAÚDE (SBCAS), 24., 2024, Goiânia/GO. Anais [...]. Porto Alegre: Sociedade Brasileira de Computação, 2024. p. 651-656. ISSN 2763-8952. DOI: https://doi.org/10.5753/sbcas.2024.2230.**

[2] TSCHANDL, P. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Harvard Dataverse, 4, 2018.

[3] SIMONYAN, K.; ZISSERMAN, A. Very deep convolutional networks for large-scale image recognition. arXiv 1409.1556, 6, 2015.

[4] ALPAYDIN, E. Introduction to machine learning. MIT Press, Cambridge, 4 edition, 2020.
