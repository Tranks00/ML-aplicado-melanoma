# ==============================================================================
# ETAPA 0: CONFIGURAÇÃO DO AMBIENTE E DEPENDÊNCIAS
# ==============================================================================
print("--- ETAPA 0: Configurando o ambiente e dependências ---")
from google.colab import files
import os

# Instala a biblioteca 'kaleido', necessária para salvar os gráficos do Plotly como imagem.
# É importante instalar no início para que o ambiente a reconheça.
!pip install -q kaleido

# Verifica se o token de API do Kaggle já existe.
if not os.path.exists('/root/.kaggle/kaggle.json'):
    # Se não existir, solicita o upload do arquivo 'kaggle.json'.
    print("Por favor, faça o upload do seu arquivo 'kaggle.json'")
    files.upload()
    
    # Cria o diretório, move o arquivo e define as permissões corretas.
    # O comando 'chmod 600' é uma medida de segurança que garante que apenas o proprietário pode ler e escrever o arquivo, protegendo suas credenciais.
    !mkdir -p ~/.kaggle
    !mv kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
else:
    print("Token do Kaggle já encontrado.")

# ==============================================================================
# IMPORTAÇÃO DAS BIBLIOTECAS PRINCIPAIS
# ==============================================================================
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import plotly.graph_objects as go

# Componentes do Scikit-learn para modelagem e avaliação.
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

print("\nTensorFlow Version:", tf.__version__)

# ==============================================================================
# DEFINIÇÃO DE FUNÇÕES AUXILIARES
# ==============================================================================

def specificity_score(y_true, y_pred):
    """
    Calcula a especificidade média para problemas multiclasse.
    A especificidade mede a proporção de negativos que foram corretamente identificados.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Calcula os verdadeiros negativos (TN) para cada classe.
    fp = cm.sum(axis=0) - np.diag(cm)  
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    
    # Calcula a especificidade por classe e retorna a média.
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp)!=0)
    return np.mean(specificity)

def extract_features(image_path, model, target_size):
    """
    Carrega uma imagem, pré-processa e extrai suas características usando um modelo pré-treinado.
    """
    try:
        img = Image.open(image_path).resize(target_size)
        img_array = np.array(img)
        
        # Garante que imagens em escala de cinza sejam convertidas para RGB (3 canais), pois a VGG19 espera essa entrada.
        if img_array.ndim == 2:
            img_array = np.stack((img_array,)*3, axis=-1)
        
        # Prepara a imagem para o formato que a rede VGG19 espera (adiciona uma dimensão de lote e normaliza os pixels).
        img_expanded = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_expanded)
        
        # Extrai o vetor de características e o "achata" para um array 1D.
        features = model.predict(img_preprocessed, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Erro ao processar a imagem {image_path}: {e}")
        return None

# ==============================================================================
# ETAPA 1: DOWNLOAD, PREPARAÇÃO E AMOSTRAGEM DOS DADOS
# ==============================================================================
print("\n--- ETAPA 1: Baixando e preparando os dados do Kaggle ---")

# Baixa o dataset completo do Kaggle e descompacta silenciosamente (-q) e sobrescrevendo (-o).
!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
print("Descompactando imagens...")
!unzip -o -q skin-cancer-mnist-ham10000.zip

# Carrega a planilha de metadados.
df_meta_completo = pd.read_csv('HAM10000_metadata.csv')

# Mapeia cada ID de imagem ao seu caminho completo no sistema de arquivos.
image_folders = ['ham10000_images_part_1', 'ham10000_images_part_2']
image_paths = {}
for folder in image_folders:
    for f in os.listdir(folder):
        if f.endswith('.jpg'):
            image_id = os.path.splitext(f)[0]
            image_paths[image_id] = os.path.join(folder, f)

df_meta_completo['path'] = df_meta_completo['image_id'].map(image_paths.get)
df_meta_completo = df_meta_completo.dropna(subset=['path'])

# Define a fração do dataset a ser usada para os experimentos.
# ATENÇÃO: Alterado para 30% conforme solicitado. A execução será mais longa.
SAMPLE_FRACTION = 0.3
print(f"\nSelecionando uma amostra de {SAMPLE_FRACTION*100}% dos dados...")

# Cria uma amostra menor do DataFrame para agilizar a execução.
# 'stratify=df_meta_completo['dx']' é crucial: garante que a proporção de cada tipo de lesão
# seja mantida na amostra, preservando as características de desbalanceamento do dataset original.
df_meta, _ = train_test_split(
    df_meta_completo, train_size=SAMPLE_FRACTION,
    stratify=df_meta_completo['dx'], random_state=42
)
print(f"Total de imagens na amostra: {len(df_meta)}")

# ==============================================================================
# ETAPA 2: EXTRAÇÃO DE ATRIBUTOS COM A REDE VGG19
# ==============================================================================
print("\n--- ETAPA 2: Extraindo atributos com VGG19 ---")
IMG_SIZE = (224, 224)

# Carrega o modelo VGG19 com pesos treinados na base ImageNet.
# 'include_top=False' remove a camada final de classificação, permitindo usar a rede apenas como um extrator de características.
base_model = VGG19(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
base_model.trainable = False # "Congela" os pesos da rede para não serem alterados durante a predição.
model_feat = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)

# Itera sobre cada imagem da amostra para extrair seus atributos. Esta é a etapa mais demorada do pipeline.
features_list = [extract_features(path, model_feat, IMG_SIZE) for path in tqdm(df_meta['path'])]

# Remove da lista qualquer imagem que possa ter falhado no processamento.
valid_indices = [i for i, f in enumerate(features_list) if f is not None]
valid_features = [features_list[i] for i in valid_indices]
df_meta = df_meta.iloc[valid_indices]

# Converte os dados para o formato NumPy, adequado para o Scikit-learn.
X = np.array(valid_features)
y_str = df_meta['dx'].values
le = LabelEncoder()
y = le.fit_transform(y_str) # Converte os rótulos de texto (ex: 'bkl') para números (ex: 0, 1, 2...).

print(f"Formato final do vetor de características (X): {X.shape}")
print(f"Formato final do vetor de rótulos (y): {y.shape}")

# ==============================================================================
# ETAPA 3: TREINAMENTO E AVALIAÇÃO DOS CLASSIFICADORES
# ==============================================================================
print("\n--- ETAPA 3: Treinando e avaliando os classificadores ---")

# Dicionário contendo os modelos de machine learning que serão avaliados.
classifiers = {
    'LDA': LinearDiscriminantAnalysis(), 'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(), 'Perceptron': Perceptron(random_state=42),
    'LR': LogisticRegression(max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42)
}

# Define o protocolo de validação cruzada 5x2, conforme o artigo.
# Isso significa que os dados serão divididos 5 vezes em 2 partes (50% treino, 50% teste),
# garantindo uma avaliação mais robusta do que uma única divisão.
cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
results = {}

# Itera sobre cada classificador para treiná-lo e avaliá-lo.
for name, clf in classifiers.items():
    print(f"\nAvaliando o classificador: {name}")
    fold_metrics = {'accuracy': [], 'precision': [], 'f1_score': [], 'specificity': []}
    
    # Executa o loop da validação cruzada (10 iterações no total).
    for train_index, test_index in tqdm(cv.split(X, y), total=10, desc=f"CV para {name}"):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calcula as métricas para esta dobra e as armazena.
        fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        fold_metrics['precision'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        fold_metrics['f1_score'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        fold_metrics['specificity'].append(specificity_score(y_test, y_pred))
    
    # Calcula a média das métricas de todas as 10 dobras para obter o resultado final do classificador.
    results[name] = {
        'Acurácia': np.mean(fold_metrics['accuracy']), 'Precisão': np.mean(fold_metrics['precision']),
        'F1-Score': np.mean(fold_metrics['f1_score']), 'Especificidade': np.mean(fold_metrics['specificity'])
    }

# ==============================================================================
# ETAPA 4: EXIBIÇÃO DOS RESULTADOS
# ==============================================================================
print("\n\n--- ETAPA 4: Resultados Finais da Validação Cruzada 5x2 ---")
df_results = pd.DataFrame.from_dict(results, orient='index')
print(df_results.round(4))

# ==============================================================================
# ETAPA 5: VISUALIZAÇÃO DOS RESULTADOS NO ESTILO DO ARTIGO
# ==============================================================================
print("\n\n--- ETAPA 5: Gerando gráfico de radar no estilo do artigo ---")

# Os eixos do gráfico serão os nomes dos CLASSIFICADORES.
labels = df_results.index

# As linhas coloridas no gráfico representarão as MÉTRICAS.
metrics_names = df_results.columns

fig = go.Figure()

# O loop itera sobre cada MÉTRICA para criar uma linha colorida no gráfico.
for metric_name in metrics_names:
    # Pega os valores de todos os classificadores para a métrica atual.
    values = df_results[metric_name].values
    
    # Adiciona a linha (trace) ao gráfico.
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=metric_name  # A legenda mostrará os nomes das métricas.
    ))

# Configura o layout final do gráfico para melhor visualização.
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]  # Mantém a escala do eixo radial de 0 a 1.
    )),
  showlegend=True,
  title="Comparativo de Desempenho dos Classificadores (Estilo Artigo)",
  # Adiciona uma margem à direita para garantir que a legenda não seja cortada.
  margin=dict(r=180) 
)

# Tenta salvar o gráfico como uma imagem. Se falhar (devido a problemas no ambiente Colab),
# ele mostrará um aviso mas continuará para exibir o gráfico interativo.
try:
    fig.write_image("grafico_radar_estilo_artigo.png")
    print("\nGráfico salvo com sucesso como 'grafico_radar_estilo_artigo.png'")
except ValueError as e:
    print(f"\nAVISO: Não foi possível salvar o gráfico como imagem. Erro: {e}")
    print("Mostrando o gráfico interativo abaixo.")

# Exibe o gráfico interativo na saída da célula.
fig.show()
