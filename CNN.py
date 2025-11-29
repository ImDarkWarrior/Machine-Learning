# ============================================
# 1. Importar librerías
# ============================================
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Detectar si hay GPU (cuda) o solo CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# ============================================
# 2. Configurar el directorio de imágenes desde Google Drive
# ============================================
# La carpeta 'Gatos' en tu Drive debe contener las subcarpetas de clases
# por ejemplo: /content/drive/MyDrive/Gatos/Gatos  y  /content/drive/MyDrive/Gatos/Random
dataset_dir = '/content/drive/MyDrive/Gatos'

# Verificar si el directorio existe
if not os.path.isdir(dataset_dir):
    raise RuntimeError(
        f"El directorio '{dataset_dir}' no existe. "
        "Asegúrate de que Google Drive esté montado y la ruta sea correcta."
    )

# Listar solo subcarpetas válidas (clases), ignorando ocultas (que empiezan con '.')
class_subfolders = [
    d for d in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')
]

# Checar que haya al menos 2 clases (gato y otros)
if len(class_subfolders) < 2:
    raise RuntimeError(
        f"No se encontraron suficientes subcarpetas de clases en '{dataset_dir}'. "
        "Debe contener al menos dos subcarpetas (ej. 'Gatos', 'Random')."
    )

print("Usaré este directorio como raíz de ImageFolder:")
print(dataset_dir)
print("Subcarpetas (clases) encontradas:", class_subfolders)

# ======================================================
# 3. Transformaciones (preprocesamiento)
#    RandomResizedCrop, RandomRotation,
#    RandomHorizontalFlip, ToTensor, Normalize
# ======================================================
# Estas transformaciones se aplican a cada imagen cuando se carga:
# - RandomResizedCrop(224): recorte aleatorio y reescalado a 224x224
# - RandomRotation(30): rotación aleatoria hasta ±30 grados
# - RandomHorizontalFlip: volteo horizontal aleatorio
# - ToTensor: convierte la imagen (PIL) a tensor PyTorch (C,H,W) en [0,1]
# - Normalize: normaliza canales RGB con medias y desviaciones típicas (tipo ImageNet)
transformaciones = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# ============================================
# 4. Cargar dataset con ImageFolder
# ============================================
# ImageFolder asume estructura:
# dataset_dir/
#    clase_0/
#        img1.jpg
#    clase_1/
#        img2.jpg
# y asigna automáticamente etiquetas 0,1... según orden alfabético
dataset = datasets.ImageFolder(root=dataset_dir,
                               transform=transformaciones)

print("Clases encontradas por ImageFolder:", dataset.classes)
print("Número total de imágenes:", len(dataset))

# -------------------------------------------------
# Diccionarios de clases
#   1) Conceptual: 0 -> gato, 1 -> otros  (pauta)
#   2) Real: índice interno PyTorch -> "gato"/"otros"
# -------------------------------------------------
# Diccionario conceptual para documentar el significado:
clases_binarias = {0: "gato", 1: "otros"}
print("Diccionario conceptual (0/1):", clases_binarias)

# class_to_idx: mapea nombre_de_carpeta -> índice entero
class_to_idx = dataset.class_to_idx  # ej. {'Gatos': 0, 'Random': 1}
label_to_name = {}
cat_idx = None

# Intentamos detectar la carpeta de "gato" por el nombre ("gat")
for cls_name, idx in class_to_idx.items():
    if 'gat' in cls_name.lower():    # si el nombre contiene "gat"
        cat_idx = idx
        label_to_name[idx] = 'gato'
    else:
        label_to_name[idx] = 'otros'

# Si no se encontró una carpeta con "gat" en el nombre,
# asumimos que la primera clase es "gato" y la segunda "otros"
if cat_idx is None:
    items_sorted = sorted(class_to_idx.items(), key=lambda x: x[1])
    label_to_name[items_sorted[0][1]] = 'gato'
    label_to_name[items_sorted[1][1]] = 'otros'

print("Mapeo índice real -> nombre:", label_to_name)

# ============================================
# 5. Dividir en entrenamiento, validación y prueba
#    70% train, 15% val, 15% test
# ============================================
len_dataset = len(dataset)
len_train   = int(len_dataset * 0.7)   # 70% para entrenamiento
len_temp    = len_dataset - len_train  # resto
len_val     = len_temp // 2            # mitad del resto para validación
len_test    = len_temp - len_val       # y el resto para test

# random_split divide el dataset en subconjuntos de forma aleatoria pero reproducible
train_ds, val_ds, test_ds = random_split(
    dataset,
    (len_train, len_val, len_test),
    generator=torch.Generator().manual_seed(42)  # semilla fija para reproducibilidad
)

print(f"Tamaño train: {len(train_ds)}")
print(f"Tamaño val  : {len(val_ds)}")
print(f"Tamaño test : {len(test_ds)}")

# ============================================
# 6. DataLoaders (ligeramente optimizados)
# ============================================
batch_size = 32
# pin_memory acelera las transferencias CPU->GPU cuando se usa CUDA
pin_memory = device.type == "cuda"

# DataLoader agrupa datos en batches y permite barajarlos y cargarlos en paralelo
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=2, pin_memory=pin_memory
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=pin_memory
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False,
    num_workers=2, pin_memory=pin_memory
)

# ============================================
# 7. Definir una CNN simple
# ============================================
# Red convolucional simple:
# Conv + ReLU + MaxPool repetidos, luego capas densas (MLP) para clasificar
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Entrada: 3 canales (RGB), salida: 16 mapas de características
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # reduce tamaño: 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 56 -> 28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),             # aplana mapas 64 x 28 x 28 → vector
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),          # apaga neuronas aleatoriamente (regularización)
            nn.Linear(128, num_classes)  # salida con num_classes logits
        )

    def forward(self, x):
        x = self.features(x)   # pasar por capas convolucionales
        x = self.classifier(x) # pasar por capas densas
        return x

# Número de clases detectadas en el dataset (debería ser 2: gato / otros)
num_classes = len(dataset.classes)
model = SimpleCNN(num_classes=num_classes)

# ============================================
# Envolver en DataParallel si hay varias GPUs
# ============================================
# Si hay más de 1 GPU, DataParallel divide los batches entre GPUs
if device.type == "cuda" and torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs con DataParallel")
    model = nn.DataParallel(model)

# Mover el modelo (o el DataParallel) al dispositivo (GPU o CPU)
model = model.to(device)

print(model)

# ============================================
# 8. Pérdida y optimizador
# ============================================
# CrossEntropyLoss es estándar para clasificación multiclase
criterion = nn.CrossEntropyLoss()
# Adam es un optimizador adaptativo, lr=1e-4 es tasa de aprendizaje
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ============================================
# 9. Funciones de entrenamiento y evaluación
# ============================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    # Poner el modelo en modo entrenamiento (activa Dropout, BatchNorm, etc.)
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        # Enviar batch al dispositivo (GPU o CPU)
        images = images.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)

        # Reiniciar gradientes
        optimizer.zero_grad()
        # Forward: salida del modelo
        outputs = model(images)
        # Calcular pérdida
        loss = criterion(outputs, labels)
        # Backpropagation
        loss.backward()
        # Actualizar pesos
        optimizer.step()

        batch_size_local = labels.size(0)
        running_loss += loss.item() * batch_size_local
        # Predicción de clase con mayor probabilidad
        _, preds = torch.max(outputs, 1)
        # Contar aciertos
        correct += (preds == labels).sum().item()
        total   += batch_size_local

    # Pérdida promedio y accuracy de la época
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    # Poner el modelo en modo evaluación (desactiva Dropout, etc.)
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    # No se calculan gradientes en evaluación (más eficiente)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size_local = labels.size(0)
            running_loss += loss.item() * batch_size_local
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += batch_size_local

    return running_loss / total, correct / total

# ============================================
# 10. Entrenamiento
# ============================================
num_epochs = 10
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Bucle principal de entrenamiento por épocas
for epoch in range(num_epochs):
    # Entrenar una época completa usando train_loader
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    # Evaluar sobre el conjunto de validación
    val_loss, val_acc     = evaluate(model, val_loader, criterion, device)

    # Guardar métricas para graficar
    train_losses.append(train_loss);  val_losses.append(val_loss)
    train_accs.append(train_acc);    val_accs.append(val_acc)

    # Mostrar resultados de la época
    print(f"Época {epoch+1}/{num_epochs} "
          f"- Loss train: {train_loss:.4f}, Acc train: {train_acc:.4f} "
          f"- Loss val: {val_loss:.4f}, Acc val: {val_acc:.4f}")

# ============================================
# 11. Gráficas de loss y accuracy
# ============================================
plt.figure(figsize=(12,4))

# Gráfica de la pérdida (loss) vs época
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train")
plt.plot(val_losses,   label="Val")
plt.title("Loss por época")
plt.xlabel("Época"); plt.ylabel("Loss")
plt.legend(); plt.grid(True)

# Gráfica de la exactitud (accuracy) vs época
plt.subplot(1,2,2)
plt.plot(train_accs, label="Train")
plt.plot(val_accs,   label="Val")
plt.title("Accuracy por época")
plt.xlabel("Época"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================
# 12. Evaluación en conjunto de prueba
# ============================================
# Evaluar desempeño final en datos nunca vistos durante entrenamiento/validación
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\nRESULTADOS EN TEST:")
print(f"Loss test: {test_loss:.4f}")
print(f"Accuracy test: {test_acc:.4f}")

# ============================================
# 13. Mostrar algunas predicciones de ejemplo
# ============================================
def imshow_tensor(img_tensor):
    # Quitar la normalización para visualizar la imagen correctamente
    img = img_tensor.clone().cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis("off")

# Seleccionar un batch del conjunto de prueba
model.eval()
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Obtener predicciones del modelo
with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

# Mostrar algunas imágenes junto con su etiqueta real y predicha
plt.figure(figsize=(10,5))
for i in range(min(8, images.size(0))):
    plt.subplot(2,4,i+1)
    imshow_tensor(images[i])
    true_label = label_to_name[int(labels[i].cpu())]
    pred_label = label_to_name[int(preds[i].cpu())]
    plt.title(f"T:{true_label}\nP:{pred_label}")
plt.tight_layout()
plt.show()
