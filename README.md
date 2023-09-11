# sirius-test-nlp-2023
Кейс по NLP для смены по ML от Тинькофф в Университете "Сириус"

### Setup

- Для начала необходимо скачать данные из чата в Telegram в формате json, добавить результирующий файл в папку data. Я выбрала чат своего потока на ПМИ ФКН ВШЭ
- Далее необходимо дообучить предобученную модель. Для этого нужно запустить все ячейки в ноутбуке ```fine_tuning.ipynb```
- Попробовать пообщаться с моделью можно в секции Inference в ноутбуке ```fine_tuning.ipynb```
- В файле ```main.py``` прописана логика чат-бота на основе дообученной модели. Пока что бот запускается только локально, его юзернейм в Telegram: @ecole_deconomie_ami_bot

### Некоторые примеры взаимодействия с ботом
![Пары](images/classes.png "Рис. 1")

![ФКН](images/activities.png "Рис. 2")

![ВМК](images/msu.png "Рис. 3")

