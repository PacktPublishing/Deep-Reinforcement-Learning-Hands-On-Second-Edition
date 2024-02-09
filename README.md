# Deep-Reinforcement-Learning
Forked from Deep-Reinforcement-Learning-Hands-On-Second-Edition, published by Packt  
Практика для курса по обучению с подкреплением. 
В отличии от оригинальных примеров кода, вместо заброшенного OpenAI gym используется поддерживаемый Farama gymnasium.   
На занятиях мы не сможем разобрать все примеры, но постараемся создать фундамент для самостоятельного освоения RL. 


## Начало работы 

КРАЙНЕ РЕКОМЕНДУЕТСЯ использовать ОС Ubuntu или Mac, т.к. библиотека gymnasium официально на Windows не поддерживается.   
Обязтаельно сделайте свой форк и дальше работайте со своим репозиторием. 
Сроки и объем выполненных заданий будут оцениваться по Вашим коммитам. 

```shell
git clone  https://github.com/yourname/Deep-Reinforcement-Learning
cd Deep-Reinforcement-Learning
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt 
```

## Семинар 1. Знакомство со средой
Пример для запуска кода
```shell
python Chapter02/01_agent_anatomy.py
```