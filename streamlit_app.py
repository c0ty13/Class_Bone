import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# Устройство для выполнения вычислений
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка обученной модели
def load_model(model_path):
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()  # Устанавливаем режим оценки
    return model

# Функция предобработки изображения
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Изменение размера изображения
        transforms.ToTensor(),          # Преобразование в тензор
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")  # Открытие изображения
    image = transform(image).unsqueeze(0)  # Добавление батч-измерения
    return image

# Функция предсказания
def predict_image(model, image_tensor):
    class_names = ['negative', 'positive']  # Метки классов
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_names[predicted.item()]
    return predicted_label

# Streamlit интерфейс
st.title("Классификация рентгеновских снимков")

# Загрузка изображения пользователем
uploaded_file = st.file_uploader("Загрузите рентгеновский снимок", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Отображение загруженного изображения
    st.image(uploaded_file, caption="Загруженное изображение", use_container_width=True)

    # Загрузка модели
    model_path = "fine_tuned_model.pth"  # Укажите путь к сохраненной модели
    model = load_model(model_path)

    # Предобработка изображения
    image_tensor = preprocess_image(uploaded_file)

    # Выполнение предсказания
    predicted_label = predict_image(model, image_tensor)

    # Вывод результата
    st.write(f"Предсказанный класс: **{predicted_label}**")
