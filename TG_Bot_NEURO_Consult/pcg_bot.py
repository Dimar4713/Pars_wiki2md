# -*- coding: utf-8 -*-
import datetime
import os
import re
import requests
import asyncio
from dotenv import load_dotenv
import json
import logging
logging.basicConfig(filename='error.log', level=logging.DEBUG)

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command, StateFilter
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.utils.callback_answer import CallbackAnswer, CallbackAnswerMiddleware
from aiogram.utils.chat_action import ChatActionSender
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State

import openai
import tiktoken
import pandas as pd

from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

import matplotlib.pyplot as plt


# Класс для управления конфигурацией бота
"""
class botConfig:
    def __init__(self):
        load_dotenv('./.env')               # Загрузка переменных окружения из файла .env
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.bot_token = os.environ.get("BOT_TOKEN")
        self.gpt_model = os.environ.get("OPENAI_MODEL")
        self.users = {}
        self.url =  ''
        self.cnk_size = 1500
        self.temperature = 0.2
        self.prompts =''
        
        # Загрузка всех переменных окружения
        self.proxy_On = os.getenv("PROXY_On")
        self.proxy_ip = os.getenv("PROXY_IP")
        self.proxy_port = os.getenv("PROXY_PORT")
        self.proxy_user = os.getenv("PROXY_USER")
        self.proxy_pass = os.getenv("PROXY_PASS")
        
        # Проверка, все ли переменные были успешно загружены из файла .env
        if None in [self.proxy_ip, self.proxy_port, self.proxy_user, self.proxy_pass, self.openai_key, self.telegram_token]:
            raise ValueError("Не все переменные окружения были загружены. Убедитесь, что все переменные указаны в файле .env")

        # Установка прокси для запросов
        if self.proxy_On == "True" :
            os.environ['HTTP_PROXY'] = f"http://{self.proxy_user}:{self.proxy_pass}@{self.proxy_ip}:{self.proxy_port}"
            os.environ['HTTPS_PROXY'] = f"http://{self.proxy_user}:{self.proxy_pass}@{self.proxy_ip}:{self.proxy_port}"        
"""
#------------------------------------------------------------------------------------------------------------------------------        

# Класс для управления конфигурацией бота
class botConfig:
    def __init__(self):
        load_dotenv('./.env')  # Загрузка переменных окружения из файла .env
        # Определение необходимых переменных окружения
        env_vars = ["OPENAI_API_KEY", "BOT_TOKEN", "GPT_MODEL", "PROXY_On", "PROXY_IP", "PROXY_PORT", "PROXY_USER", "PROXY_PASS"]
        for var in env_vars:   setattr(self, var.lower(), os.getenv(var))

        self.users = {}
        self.url = ''
        self.cnk_size = 1500
        self.temperature = 0.2
        self.prompts = ''

        # Проверка наличия всех необходимых переменных окружения
        if self.proxy_on == "True":
            if not all([getattr(self, var.lower()) for var in env_vars]):
                raise ValueError("Не все переменные окружения были загружены. Убедитесь, что все переменные указаны в файле .env")
            # Установка прокси для запросов, если включено
            proxy = f"http://{self.proxy_user}:{self.proxy_pass}@{self.proxy_ip}:{self.proxy_port}"
            os.environ['HTTP_PROXY'] = os.environ['HTTPS_PROXY'] = proxy

#------------------------------------------------------------------------------------------------------------------------------        

    def get_user(self, user_id):
        if not user_id in self.users:
            self.prompts = {}
            with open('prompts.json', 'r', encoding='utf-8') as file:
              self.prompts = json.load(file)

            self.users[user_id] = {
                'openai_api_key': self.openai_api_key if user_id ==  201173615 else '',
                'gpt_model': self.gpt_model,
                'prompts' : self.prompts,
                'url': self.url,
                'temperature': self.temperature,
                'cnk_size': self.cnk_size,
                'verbose': 0,
                'history': [],
                'summary': ''
            }
            print("New user: ", user_id)

        return self.users[user_id]

    def set_user_param(self, user_id, param, value):
        self.users[user_id][param] = value
#------------------------------------------------------------------------------------------------------------------------------        

# Класс для взаимодействия с OpenAI
class OpenAIIntegration:
    def __init__(self, botConfig):
        self.config = botConfig
        self.ai = openai
        self.ai.openai_api_key = self.config['openai_api_key']
        self.gpt_model = self.config['gpt_model']
        self.prompts = self.config['prompts']
        self.chunks = TextProcessor(self.config)
        self.prices = {
            'gpt-3.5-turbo-16k': {
                'prompt': 0.001,
                'completion': 0.002
            },
            'gpt-3.5-turbo': {
                'prompt': 0.0015,
                'completion': 0.002
            },
            'gpt-3.5-turbo-0301': {
                'prompt': 0.0015,
                'completion': 0.002
            },
            'gpt-4': {
                'prompt': 0.03,
                'completion': 0.06
            },
            'gpt-4-1106-preview': {
                'prompt': 0.01,
                'completion': 0.03
            }
        }

    def generate_response(self, prompt):

        

        # Prepeare chunks
        chunks = self.chunks.get_rel_chunks(prompt, 3)

        lvl = 0.5

        knowledge = ""
        for index, (chunk, score) in enumerate(chunks):
            if score <= lvl:
                print(score, len(chunk.page_content))
                knowledge += f"Чанк из БЗ #{index} Релевантность:{round(score, 5)}\n{chunk.page_content}\n\n"

        if knowledge == '':
            knowledge = "Common knowledge"

        print(len(chunks))
        history = ""
        for dialog in self.config['history'][-5:]:
            history += f'\n{dialog}\n'

        """
        Генерирует ответ на заданный запрос с использованием OpenAI GPT.
        """

        system_text = f'{self.prompts}'
                
                # Сообщений чата
        chat_messages = [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": f"Please answer the user's question \"{prompt}\" using the information from the knowledge base \"{knowledge}\". Aim for a complete and thorough response, focusing on relevant details only. If the knowledge base lacks specific information for the user's question, craft an answer based on your expertise but clearly indicate the source of your knowledge. Avoid explicit references to documents or the database. Convey to the user that the knowledge base's information is integrated into your understanding. Respond in the user's preferred language; default to Russian if unspecified."
            }
        ]                



        try:
            response = self.ai.ChatCompletion.create(
                model=self.gpt_model,
                messages=chat_messages,
                temperature=float(self.config['temperature'])
            )
            # Generate UUID
            current_datetime = datetime.datetime.now()
            id = int(f"{current_datetime.strftime('%Y%m%d%H%M%S')}{current_datetime.microsecond}")

            answer = response.choices[0].message['content'].strip()
            dialog = {"id": id, "model": self.gpt_model, "tokens_prompt": response.usage["prompt_tokens"],
                      "tokens_total": response.usage["total_tokens"], "price_usd": round(
                    self.get_price(response.usage["prompt_tokens"], response.usage["completion_tokens"]), 5),
                      "question": prompt, "answer": answer, "score": None, "chunks": knowledge}
            self.config['history'].append(dialog)
            # self.config['summary'] = self.summarize_dialog(prompt, answer)
            info = ''
            if self.config['verbose']:
                info = f'''
Модель: {self.gpt_model.upper()}                
Токены (Промпт/Ответ/Всего): {response.usage["prompt_tokens"]}/{response.usage["completion_tokens"]}/{response.usage["total_tokens"]}
Цена: ${round(self.get_price(response.usage["prompt_tokens"], response.usage["completion_tokens"]), 5)} / {round(self.get_price(response.usage["prompt_tokens"], response.usage["completion_tokens"]) * 90, 2)} руб. 
Использование БЗ:
{knowledge}
                '''

            return [id, answer, info]
        except openai.error.OpenAIError as e:
            # Обработка возможных ошибок от API
            print(f"Произошла ошибка при обращении к OpenAI: {e}")
            return None

    def get_price(self, prompt, compl):
        modelPrice = self.prices[self.gpt_model]
        price = (prompt * modelPrice['prompt'] + compl * modelPrice['completion']) / 1000
        return price

    def summarize_dialog(self, q, a):

        max_tokens = 1000
        temperature = 0

        system_text = '''
        Ты профессиональный доктор который качественно ведет историю болезней и диалогов с пациентами.
        Твоя задача суммаризировать вопрос и ответ с предыдущей суммаризацией, для последующего понимания о чем ведется диалог с клиентом.
        Ты обязан как минимум держать смысл последних 5 сообщений. 
        '''

        # Сообщений чата
        chat_messages = [
            {"role": "system", "content": system_text},
            {"role": "user",
             "content": f"Предыдущая суммаризация: {self.config['summary']}\n\Новый вопрос от пациента: \n{q}\n\n Вот ответ который пациент получил: \n{a}"}
        ]

        try:
            response = self.ai.ChatCompletion.create(
                model=self.gpt_model,
                messages=chat_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message['content'].strip()
        except openai.error.OpenAIError as e:
            # Обработка возможных ошибок от API
            print(f"Произошла ошибка при суммаризации: {e}")
#------------------------------------------------------------------------------------------------------------------------------        

# Класс для обработки текста
class TextProcessor:
    def __init__(self, botConfig):
        # Здесь можно инициализировать любые необходимые переменные
        self.embeddings = OpenAIEmbeddings()
        if os.path.exists("faiss"):
            print(f"Загружаем векторную базу ФАИС!")

            self.db = FAISS.load_local("faiss", OpenAIEmbeddings())

        else:
            knowledge_base = botConfig['url']
            if (('google' in knowledge_base) and ('document' in knowledge_base)):
                self.database = self.load_document_text(knowledge_base)
            elif ('google' in knowledge_base): 
                 knowledge_base= self.convert_drive_link_to_direct_download(knowledge_base)
                 self.database = self.download_doc(knowledge_base)
            else:
                self.database = self.download_doc(knowledge_base)


            # Очистить от двойных переносов возможно надо!
            self.chunks = self.split_text(self.database, botConfig['cnk_size'])

            # Создание индекса в FAISS
            print(f"Создаем векторную базу FAISS!")
            self.db = FAISS.from_documents(self.chunks, self.embeddings)
            self.db.save_local("faiss")

    def convert_drive_link_to_direct_download(self,link):
            # Проверяем, является ли ссылка ссылкой Google Drive
            if "drive.google.com" in link:
                # Извлекаем ID файла из ссылки
                file_id = link.split('/d/')[1].split('/view')[0]
                # Формируем прямую ссылку для скачивания
                direct_download_link = f"https://drive.google.com/uc?export=download&id={file_id}"
                return direct_download_link
            else:
                return "Это не ссылка Google Drive."
            
    def get_rel_chunks(self, question, chunks_amount=5, vdb="F"):
        if vdb == "F":
            chunks = self.db.similarity_search_with_score(question, k=chunks_amount)
        else:
            # Поиск похожих фрагментов в Qdrant
            chunks = self.qdrant.similarity_search_with_score(question, k=chunks_amount)

        return chunks

    def get_embedding(self, text):
        embedding = self.embeddings.encode(text)  # Получение вектора для фрагмента текста
        return embedding

    def text_to_markdown(text):
        """
        Преобразует обычный текст в формат Markdown.
        """

        def replace_header(match, level):
            header = '#' * level
            return f"{header} {match.group(2)}\n{match.group(2)}"

        text = re.sub(r'^(I{1,3}|IV|V)\. (.+)', lambda m: replace_header(m, 1), text, flags=re.M)
        text = re.sub(r'\*([^\*]+)\*', lambda m: replace_header(m, 2), text)

        return text

    def load_document_text(self, url: str) -> str:
        """
        Загружает текст из документа Google Docs по URL.
        """
        match_ = re.search(r'/document/d/([a-zA-Z0-9-_]+)', url)
        if not match_:
            raise ValueError('Invalid Google Docs URL')

        doc_id = match_.group(1)
        google_docs_url = f'https://docs.google.com/document/d/{doc_id}/export?format=txt'

        return self.download_doc(google_docs_url)

    


    def download_doc(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise ConnectionError(f"Ошибка при загрузке документа: {e}")

    def num_tokens_from_string(self, string: str, encoding_name: str = 'cl100k_base') -> int:
        """Возвращает количество токенов в строке"""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def split_text(self, text, max_count):
        headers_to_split_on = [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
            ("####", "Header4"),
            ("#####", "Header5"),
            ("######", "Header6"),
            ("#######", "Header7"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        fragments = markdown_splitter.split_text(text)

        Max_Headers = 6

        for i, chunk in enumerate(fragments, start=0):
            # Объедините метаданные и содержимое страницы
            if hasattr(chunk, 'metadata') and hasattr(chunk, 'page_content'):
                # if 'metadata' in chank and 'page_content' in chank:
                meta_data = chunk.metadata
                # Собираем значения всех полей в строку через запятую
                values = [meta_data.get(f'Header{i}', '') for i in range(1, Max_Headers + 1)]
                csv_string = ', '.join(values)
                page_content = chunk.page_content
                combined_content = f"{csv_string}\n{page_content}"
                # Перезапись поля page_content
                chunk.page_content = combined_content

        return fragments

    def create_index_database(self, documents):
        if not hasattr(self, 'embeddings'):
            self.create_embeddings_model()
        self.db = FAISS.from_documents(documents, OpenAIEmbeddings())

#=====================================================================================================================================================
#!!!!!!!!               Основной класс бота          !!!!!!!!!!!!!!!!    
#=====================================================================================================================================================
class ConsultDocBot:
    def __init__(self, bot_config):
        self.config = bot_config

        self.bot = Bot(token=self.config.bot_token)
        self.dp = Dispatcher()
        self.dp.callback_query.middleware(CallbackAnswerMiddleware())
        self.parse_mode = ""
        self.available_models = ["gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-3.5-turbo-0301", "gpt-4",
                                 "gpt-4-1106-preview"]

        self. waiting_for_db_path = State()  # Состояние для запроса пути к базе данных  #16/02/24
        self.waiting_for_key = State()
        self.waiting_for_model = State()
        self.waiting_for_temperature = State()
        self.waiting_for_verbose = State()
        
        #self.waiting_for_db_path = State()  # Состояние для ожидания пути к базе знаний
        # start - Начало работы бота
        # setup - Настройки
        # help - Помощь
        # showgraph - График чанков
        # summary - Суммаризация (не активно)
        # history - История

        # Регистрация обработчиков команд
        @self.dp.message(CommandStart())
        async def command_start_handler(message: Message) -> None:

            cfg = self.userConfig(message.from_user.id)

            # 'api_key': self.api_key,
            # 'gpt_model': self.gpt_model,
            # 'url': self.url,
            # 'temperature': self.temperature,
            # 'cnk_size': self.cnk_size,
            # 'patient': patient,
            # 'verbose': 0,
            # 'history': [],
            # 'summary': ''


            print(len(self.config.users))
            for user, params in self.config.users.items():
                print(
                    f"User:{user} => {params['openai_api_key']}, {params['gpt_model']}, {params['temperature']}, {params['verbose']}")

            # Создаем клавиатуру с кнопками

            print(message)
            await message.answer('Приветствую Вас! Это бот нейро-консльтант по документации и базе знаний компании',
                                 parse_mode=self.parse_mode)

        @self.dp.message(Command(commands=['setup']))
        async def command_setup_handler(message: Message, state: FSMContext) -> None:

            cfg = self.userConfig(message.from_user.id)

            file_idx = "./faiss/index.faiss"
            modification_date = ""
            if os.path.exists(file_idx):
                timestamp = os.path.getmtime(file_idx)
                modification_date = datetime.datetime.fromtimestamp(timestamp)

            try:
                t = modification_date.strftime('%H:%M:%S %d.%m.%Y')
            except:
                t = "----"

            msg = f"""
⚙️Настройки бота:
    🕹Ключ: {'*' * 7 + cfg['openai_api_key'][-5:]}
    🕹Модель: {cfg['gpt_model']}
    🕹Температура: {cfg['temperature']}
    🕹Тех. инфо (verbose): {"✅" if cfg['verbose'] > 0 else "❌"}
    🕹Векторная БД обновлена (UTC): {t}   
            """

            # Создаем клавиатуру с кнопками
            builder = InlineKeyboardBuilder()
            builder.add(
                types.InlineKeyboardButton(
                    text="Ключ OpenAI",
                    callback_data="set_new_openai_key"
                )
            )
            builder.add(
                types.InlineKeyboardButton(
                    text="Модель",
                    callback_data="change_model"
                )
            )
            builder.add(
                types.InlineKeyboardButton(
                    text="Температура",
                    callback_data="change_temperature"
                )
            )
            builder.add(
                types.InlineKeyboardButton(
                    text="Тех. инфо",
                    callback_data="change_verbose"
                )
            )
            builder.add(
                types.InlineKeyboardButton(
                    text="Пересоздать БЗ",
                    #callback_data="update_vector_db" request_db_path
                    callback_data="request_db_path"
                    
                )
            )
            builder.add(
                types.InlineKeyboardButton(
                    text="Скачать историю",
                    callback_data="download_history"
                )
            )
            builder.adjust(2)

            print(message)
            await message.answer(msg, reply_markup=builder.as_markup(), parse_mode=self.parse_mode)
            # Устанавливаем пользователю состояние "выбирает название"
#----------------------------------------------------------------------------------------------------------
# Блок замены ключа            
        @self.dp.callback_query(F.data == "set_new_openai_key")
        async def ask_openai_key(callback: CallbackQuery, callback_answer: CallbackAnswer, state: FSMContext):
            await callback.message.delete()
            await callback.message.answer(f"Отправьте ваш OpenAI ключ", parse_mode=self.parse_mode)
            # Устанавливаем пользователю состояние "waiting_for_key"
            await state.set_state("waiting_for_key")

        @self.dp.message(StateFilter("waiting_for_key"))
        async def save_openai_key(message: Message, state: FSMContext):

            self.config.set_user_param(message.from_user.id, 'openai_api_key', message.text.strip())
            self.openai_integration = ''

            # await message.delete(message.message_id-1)
            await message.delete()

            await state.clear()  # Завершаем состояние
            await message.answer("Ключ сохранен.")
#----------------------------------------------------------------------------------------------------------
        @self.dp.callback_query(F.data == "change_model")
        async def ask_model(callback: CallbackQuery, callback_answer: CallbackAnswer, state: FSMContext):

            builder = InlineKeyboardBuilder()
            for index, model in enumerate(self.available_models):
                builder.add(
                    types.InlineKeyboardButton(
                        text=model.upper(),
                        callback_data=model
                    )
                )
            builder.adjust(3)

            await callback.message.delete()

            await callback.message.answer(f"Выберите модель", reply_markup=builder.as_markup(),
                                          parse_mode=self.parse_mode)
            # Устанавливаем пользователю состояние "waiting_for_key"
            await state.set_state("waiting_for_model")

        @self.dp.message(StateFilter("waiting_for_model"))
        @self.dp.callback_query(lambda c: c.data in self.available_models)
        async def save_model(callback: CallbackQuery, callback_answer: CallbackAnswer, state: FSMContext):

            print(callback.data)
            self.config.set_user_param(callback.from_user.id, 'gpt_model', callback.data)
            self.openai_integration = ''

            print(callback.message.message_id)
            await callback.message.delete()

            await state.clear()  # Завершаем состояние
            await callback.message.answer("Модель изменена.")
#----------------------------------------------------------------------------------------------------------
        @self.dp.callback_query(F.data == "change_temperature")
        async def ask_temperature(callback: CallbackQuery, callback_answer: CallbackAnswer, state: FSMContext):
            await callback.message.delete()
            await callback.message.answer(f"Отправьте значение температуры, должно быть число больше 0.01 и до 1.00",
                                          parse_mode=self.parse_mode)
            # Устанавливаем пользователю состояние "waiting_for_key"
            await state.set_state("waiting_for_temperature")

        @self.dp.message(StateFilter("waiting_for_temperature"))
        async def save_temperature(message: Message, state: FSMContext):

            self.config.set_user_param(message.from_user.id, 'temperature', message.text.strip())
            self.openai_integration = ''

            # await message.delete(message.message_id-1)
            await message.delete()

            await state.clear()  # Завершаем состояние
            await message.answer("Температура изменена.")
#----------------------------------------------------------------------------------------------------------
        @self.dp.callback_query(F.data == "change_verbose")
        async def ask_verbose(callback: CallbackQuery, callback_answer: CallbackAnswer, state: FSMContext):

            builder = InlineKeyboardBuilder()
            builder.add(
                types.InlineKeyboardButton(
                    text="✅ Вкл",
                    callback_data="set_verbose_1"
                )
            )
            builder.add(
                types.InlineKeyboardButton(
                    text="❌ Выкл",
                    callback_data="set_verbose_0"
                )
            )
            builder.adjust(2)

            await callback.message.delete()

            await callback.message.answer(f"Техническую информацию?", reply_markup=builder.as_markup(),
                                          parse_mode=self.parse_mode)
            # Устанавливаем пользователю состояние "waiting_for_verbose"
            await state.set_state("waiting_for_verbose")

        @self.dp.message(StateFilter("waiting_for_verbose"))
        @self.dp.callback_query(lambda c: 'set_verbose_' in c.data)
        async def save_model(callback: CallbackQuery, callback_answer: CallbackAnswer, state: FSMContext):

            print(callback.data)
            self.config.set_user_param(callback.from_user.id, 'verbose', int(callback.data[-1]))
            self.openai_integration = ''

            print(callback.message.message_id)
            await callback.message.delete()

            await state.clear()  # Завершаем состояние
            await callback.message.answer("Параметр настройки изменен.")
#----------------------------------------------------------------------------------------------------------
# Блок запроса пути к БЗ
        @self.dp.callback_query(F.data == "request_db_path")
        async def ask_db_path(callback: CallbackQuery, callback_answer: CallbackAnswer, state: FSMContext):
            await callback.message.delete()
            await callback.message.answer(f"Пожалуйста, укажите путь к базе знаний:", parse_mode=self.parse_mode)
            # Устанавливаем пользователю состояние "waiting_for_path"
            await state.set_state("waiting_for_path")

        @self.dp.message(StateFilter("waiting_for_path"))
        async def update_db_path(message: Message, state: FSMContext):
            cfg = self.userConfig(message.from_user.id)

            # Проверка наличия установленного ключа API в конфигурации пользователя
            if not cfg.get('openai_api_key'):
                await message.answer("Ошибка: ключ API не установлен. Пожалуйста, установите ключ API с помощью соответствующей команды.")
                await state.clear()  # Завершаем состояние без выполнения оставшейся части функции
                return  # Выход из функции

            self.config.set_user_param(message.from_user.id, 'url', message.text.strip())
            #self.db_path = ''
            self.db_path = message.text.strip()
            # await message.delete(message.message_id-1)
            await message.delete()

            await state.clear()  # Завершаем состояние
            await message.answer(f"Путь сохранен сохранен =>")

            cfg = self.userConfig(message.from_user.id)
            db_path = cfg['url']
            try:
                folder = db_path  # Использование полученного от пользователя пути

                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder, filename)
                        os.remove(file_path)
                    os.rmdir(folder)

                TextProcessor(cfg)
                msg = "Векторная база обновлена!"
            except Exception as e:
                print(e)  # Вывод информации об ошибке в консоль для отладки
                msg = "Векторная база обновление: Что-то пошло не так. Попробуйте позже!"

            await message.answer(msg)
            print(msg)

            # Сброс состояния
            # Для aiogram 3.x и выше
            await state.set_state(None)
            #await state.finish()

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
        """  
        @self.dp.callback_query(F.data == "request_db_path")
        async def request_db_path(callback: CallbackQuery):
            # Сообщение пользователю с просьбой указать путь к базе знаний
            await callback.message.answer("Пожалуйста, укажите путь к базе знаний:")

            # Регистрация следующего состояния, ожидающего ответа от пользователя
            await self.waiting_for_db_path.set()

        @self.dp.message(state=self.waiting_for_db_path)
        async def update_vector_db(message: Message, state: FSMContext):
            # Получение пути к базе знаний от пользователя
            db_path = message.text

            cfg = self.userConfig(message.from_user.id)

            try:
                folder = db_path  # Использование полученного от пользователя пути

                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder, filename)
                        os.remove(file_path)
                    os.rmdir(folder)

                TextProcessor(cfg)
                msg = "Векторная база обновлена!"
            except Exception as e:
                print(e)  # Вывод информации об ошибке в консоль для отладки
                msg = "Векторная база обновление: Что-то пошло не так. Попробуйте позже!"

            await message.answer(msg)
            print(msg)

            # Сброс состояния
            await state.finish()
        """




        """
        @self.dp.callback_query(F.data == "update_vector_db")
        async def update_vector_db(callback: CallbackQuery, callback_answer: CallbackAnswer):

            cfg = self.userConfig(callback.from_user.id)

            try:
                folder = "faiss"

                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder, filename)
                        os.remove(file_path)
                    os.rmdir(folder)

                TextProcessor(cfg)
                msg = "Векторная база обновлена!"
                callback_answer.text = msg
            except:
                msg = "Векторная база обновление: Что-то пошло не так. Попробуйте позже!"
                callback_answer.text = msg
                callback_answer.cache_time = 10
            await callback.message.answer(msg)
            print(msg)            
            """
 #----------------------------------------------------------------------------------------------------------           
        @self.dp.callback_query(F.data == "download_history")
        async def download_history_xls(callback: CallbackQuery, callback_answer: CallbackAnswer):

            cfg = self.userConfig(callback.from_user.id)

            if not len(cfg["history"]):
                callback_answer.text = "История пустая!"
                callback_answer.cache_time = 10
                return

            # Создаем DataFrame из данных
            df = pd.DataFrame(cfg["history"])

            # Создаем Excel-файл (xlsx)
            file_name = "output.xlsx"
            df.to_excel(file_name, index=False)

            print(f"Создан Excel-файл: {file_name}")

            try:

                input_file = FSInputFile(path=file_name, filename="history.xlsx")
                await self.bot.send_document(chat_id=callback.message.chat.id, document=input_file,
                                             caption="Excel-файл с историей")

                # Удаляем файл после отправки
                os.remove("output.xlsx")

                # Устанавливаем ответ для callback_query
                msg = "Excel-файл с историей отправлен."
                callback_answer.text = msg

            except:
                msg = "Подготовка файла: Что-то пошло не так. Попробуйте позже!"
                callback_answer.text = msg
                callback_answer.cache_time = 10
                await callback.message.answer(msg)

            print(msg)
#----------------------------------------------------------------------------------------------------------
        ### Команды!!!

        @self.dp.message(Command(commands=['help']))
        async def command_help_handler(message: types.Message):
            cfg = self.userConfig(message.from_user.id)
            msg = f'''
/start - Начало работы бота
/setup - Настройки
/help - Помощь
/showgraph - График чанков
/summary - Суммаризация (не активно)
/history - История
            '''
            await message.answer(f"{msg}", parse_mode=self.parse_mode)

        @self.dp.message(Command(commands=['summary']))
        async def command_summary_handler(message: types.Message):
            cfg = self.userConfig(message.from_user.id)
            await message.answer(f"Summary:\n{cfg['summary']}", parse_mode=self.parse_mode)

        @self.dp.message(Command(commands=['history']))
        async def command_history_handler(message: types.Message):
            cfg = self.userConfig(message.from_user.id)
            history = ""
            for dialog in cfg['history'][-5:]:
                history += f'\nВопрос: {dialog["question"]}\nОтвет: {dialog["answer"]}\nОценка: {dialog["score"]}\n'

            parts = [history[i:i + 1000] for i in range(0, len(history), 1000)]
            await message.answer(f'History:', parse_mode=self.parse_mode)
            for part in parts:
                await message.answer(f'{part}', parse_mode=self.parse_mode)

        @self.dp.message(Command(commands=['showgraph']))
        async def command_showgraph_handler(message: types.Message):


            db = FAISS.load_local("faiss", OpenAIEmbeddings())
            showchunks = 0
            gstart = 0
            gend = 10000
            parsed_msg = message.text.split(" ")
            for idx in parsed_msg:
                p = idx.split("=")
                if p[0] == "gstart":
                    gstart = int(p[1])
                if p[0] == "gend":
                    gend = int(p[1])
                if p[0] == "showchunks":
                    showchunks = int(p[1])


            sizes = []
            samples = []
            t = TextProcessor(self.config)
            for doc in db.docstore._dict:
                size = t.num_tokens_from_string(db.docstore._dict[doc].page_content)
                # size = len(db.docstore._dict[doc].page_content)
                if size > gstart and size < gend:
                    sizes.append(size)
                    if showchunks and size < showchunks:
                        samples.append(db.docstore._dict[doc].page_content)


            # print(sizes)
            plt.hist(sizes, bins=100)
            plt.title("Распределение размеров документов")
            plt.xlabel("Размер документа (токены!)")
            plt.ylabel("Число документов")

            plt.tight_layout()
            plt.savefig("graph.png")
            # plt.show()

            photo = FSInputFile(r'graph.png')
            await message.answer_photo(photo, caption=f'Graph (всего: {len(sizes)})')
            if showchunks:
                lst = '\n'.join(samples)
                await message.answer(f'''
Всего найдено кусков с размером до {showchunks} символов: {len(samples)}
{lst[:4000]}             
                ''')

       
        # Регистрация обработчика текстовых сообщений
        @self.dp.message()
        async def message_handler(message: types.Message):

            async with ChatActionSender.typing(chat_id=message.chat.id, bot=self.bot):

                cfg = self.userConfig(message.from_user.id)
                if not (self.checkConfig(cfg) or self.openai_integration):
                    await message.answer(
                        f'Ответ нейро-помощника:\nУ вас не настроены ключ/модель!\nУстановка ключа нажмите /openai_key\nУстановка модели',
                        parse_mode=self.parse_mode)
                    return

                user_text = message.text
                builder = InlineKeyboardBuilder()

                try:
                    openai_integration = OpenAIIntegration(cfg)
                    msg_id, response, info = openai_integration.generate_response(user_text)
                    builder.add(
                        types.InlineKeyboardButton(
                            text="-2",
                            callback_data=f"set_answer_score_0_{msg_id}"
                        )
                    )
                    builder.add(
                        types.InlineKeyboardButton(
                            text="-1",
                            callback_data=f"set_answer_score_1_{msg_id}"
                        )
                    )
                    builder.add(
                        types.InlineKeyboardButton(
                            text="0",
                            callback_data=f"set_answer_score_2_{msg_id}"
                        )
                    )
                    builder.add(
                        types.InlineKeyboardButton(
                            text="+1",
                            callback_data=f"set_answer_score_3_{msg_id}"
                        )
                    )
                    builder.add(
                        types.InlineKeyboardButton(
                            text="+2",
                            callback_data=f"set_answer_score_4_{msg_id}"
                        )
                    )
                    
                    
                except Exception as e:
                    print("ERROR! {e}")
                    logging.exception("Произошла ошибка при обращении к OpenAI")
                    response = f"Произошла ошибка!\n{e}"
                    info = ""

                await message.answer(f'Ответ нейро-помощника:\n{response}', reply_markup=builder.as_markup(),
                                     parse_mode=self.parse_mode)
                if len(info) > 0:
                    await message.answer(f'Техническая информация:', parse_mode=self.parse_mode)
                    parts = [info[i:i + 1000] for i in range(0, len(info), 1000)]
                    for part in parts:
                        await message.answer(f'{part}', parse_mode=self.parse_mode)

            @self.dp.callback_query(lambda c: "set_answer_score_" in c.data)
            async def save_model(callback: CallbackQuery, callback_answer: CallbackAnswer, state: FSMContext):

                cfg = self.userConfig(callback.from_user.id)
                data = callback.data.split("_")
                id = int(data[-1])
                score = int(data[-2]) - 2

                for dialog in cfg["history"]:
                    if dialog["id"] == id:
                        dialog.update({"score": score})
                self.config.set_user_param(callback.from_user.id, 'history', cfg["history"])

                callback_answer.text = f"Данному ответу вы установили оценку: {'+' if score > 0 else ''}{score}"

            # Others
            @self.dp.callback_query()
            async def cb_handler(callback: CallbackQuery, callback_answer: CallbackAnswer):

                if True:
                    callback_answer.text = "Отлично!"
                else:
                    callback_answer.text = "Что-то пошло не так. Попробуйте позже"
                    callback_answer.cache_time = 10

    class ChangeKey(StatesGroup):
        waiting_for_key = State()  # Состояние ожидания ввода ключа

    def userConfig(self, user_id):
        cfg = self.config.get_user(user_id)
        return cfg

    

    def checkConfig(self, cfg):
        # 'api_key': self.api_key,
        # 'gpt_model': self.gpt_model,
        # 'url': self.url,
        # 'temperature': self.temperature,
        # 'cnk_size': self.cnk_size,
        # 'verbose': 0,
        # 'history': [],
        # 'summary': ''
        return cfg['openai_api_key'] or cfg['gpt_model']

    async def run(self):
        print("Starting  Bot...")
        await self.dp.start_polling(self.bot, skip_updates=True)
#=====================================================================================================================================================
# Конец  class ConsultDocBot
#=====================================================================================================================================================


# Пример использования класса
if __name__ == "__main__":
    logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
    logging.getLogger("chromadb").setLevel(logging.ERROR)

    config = botConfig()
    userid= 5773050493
    user=config.get_user( userid)
    print(config.users)

    bot = ConsultDocBot(config)
    asyncio.run(bot.run())
