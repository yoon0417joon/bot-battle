import math, random
import pyfiglet  # required_libraries에 명시된 라이브러리

# 봇 데코레이터 정의
def bot_decorator(name, description):
    def decorator(func):
        def wrapper(input_value, *args, **kwargs):
            # 입력값이 문자열이면 메타데이터 반환
            if isinstance(input_value, str):
                if input_value.lower() == "name":
                    return name
                elif input_value.lower() == "description":
                    return description
            # 그 외의 경우 원래 로직 실행
            return func(input_value, *args, **kwargs)
        wrapper.bot_name = name
        wrapper.bot_description = description
        return wrapper
    return decorator

# 필요한 라이브러리 목록
required_libraries = ["pyfiglet"]

addon_variable = 42

@bot_decorator("Figlet Bot", "This bot uses pyfiglet to render its name in ASCII art.")
def mybot1(whichbotitis):
    # 숫자 입력이면 기존 로직 실행 (pyfiglet를 이용하여 'trust' 텍스트 출력)
    art = pyfiglet.figlet_format("trust")
    print(art)
    return "trust"

@bot_decorator("Random Bot", "This bot behaves randomly and sometimes surprises you.")
def mybot2(whichbotitis):
    # 숫자 입력이면 확률에 따라 'betray' 또는 'trust' 반환
    return "betray" if random.random() < 0.5 else "trust"

# 예외 함수 (봇으로 처리되지 않음)
def helper_function(x):
    return math.sqrt(x)

# 봇 함수로 사용할 함수 이름 목록 (데코레이터를 적용한 함수의 실제 이름과 일치해야 함)
bots = ["mybot1", "mybot2"]

# 봇으로 처리하지 않을(일반) 함수 이름 목록
exceptions = ["helper_function"]
