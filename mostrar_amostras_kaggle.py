import os
import matplotlib.pyplot as plt
import cv2

# Caminho para as pastas (use r'' para lidar com as barras invertidas do Windows)
base_dir = r'C:\Users\mulle\Documents\tumor_classification\Brain Tumor MRI Dataset\Training'
classes = ['notumor', 'glioma', 'meningioma', 'pituitary']

# Tamanho das imagens para exibição
IMG_SIZE = 100

fig, axes = plt.subplots(1, 4, figsize=(15, 5))

for i, categoria in enumerate(classes):
    caminho_classe = os.path.join(base_dir, categoria)
    imagem_nome = os.listdir(caminho_classe)[0]  # pega a primeira imagem da pasta
    caminho_imagem = os.path.join(caminho_classe, imagem_nome)
    
    img = cv2.imread(caminho_imagem)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    axes[i].imshow(img)
    axes[i].set_title(categoria.capitalize())
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('figura1_amostras_kaggle.png')  # Salva a imagem no mesmo diretório
plt.show()

