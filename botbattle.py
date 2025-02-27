import sys
import os
import math
import time
import random
import itertools
import inspect
import statistics
import importlib.util
import textwrap

# matplotlib, mplcursors
import mplcursors
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QComboBox, QRadioButton,
    QCheckBox, QTabWidget, QTextEdit, QScrollArea,
    QFileDialog, QMessageBox, QButtonGroup, QFrame, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QTimer, QEvent
from PyQt6.QtGui import QPixmap
from qt_material import apply_stylesheet

# 전역 변수들
b1log = []      # 1번 봇 행동 기록
b2log = []      # 2번 봇 행동 기록
b1play = "none"
b2play = "none"
b1point = 0     # 1번 봇 누적 점수
b2point = 0     # 2번 봇 누적 점수
round = 0       # 현재 라운드
tester_retribution = 0
tester_decision = "none"
addon_image_paths = {}

excluded = {"battle", "main", "getCombinedBotList", "bot_decorator", "apply_stylesheet", "load_bot_image", "morecommon"}
all_functions = []
bot_functions = []
total_bot_count = 0
selected_bots_global = []
overall_scores_global = {}
original_bots = []
addons_bots = {}

def bot_decorator(name, description):
    def decorator(func):
        def wrapper(input_value, *args, **kwargs):
            # 입력이 문자열이면 이름 또는 설명 반환
            if isinstance(input_value, str):
                if input_value.lower() == "name":
                    return name
                elif input_value.lower() == "description":
                    return description
            # 숫자 등 다른 입력이면 원래 로직 실행
            return func(input_value, *args, **kwargs)
        # 메타데이터를 속성으로 추가 (옵션)
        wrapper.bot_name = name
        wrapper.bot_description = description
        return wrapper
    return decorator

def morecommon(lst):
    count1 = count2 = 0
    for val in lst:
        if val == "trust":
            count1 += 1
        elif val == "betray":
            count2 += 1
    if count1 > count2:
        return "trust"
    elif count2 > count1:
        return "betray"
    else:
        return 0

def battle(b1_name, b2_name, rounds, print_result=True):
    global round, b1point, b2point, b1log, b2log
    bot1 = globals()[b1_name]
    bot2 = globals()[b2_name]
    round = 0
    b1point = 0
    b2point = 0
    b1log.clear()
    b2log.clear()
    for i in range(rounds):
        b1play = bot1(1)
        b2play = bot2(2)
        if b1play == "trust" and b2play == "trust":
            b1point += 3
            b2point += 3
        elif b1play == "betray" and b2play == "trust":
            b1point += 5
        elif b1play == "trust" and b2play == "betray":
            b2point += 5
        elif b1play == "betray" and b2play == "betray":
            b1point += 1
            b2point += 1
        b1log.append(b1play)
        b2log.append(b2play)
        round += 1
    b1log.clear()
    b2log.clear()
    if print_result:
        print(f"Final Score: {b1_name}({b1point}) vs {b2_name}({b2point})")
    return {b1_name: b1point, b2_name: b2point}

# 봇 이미지 로딩 함수 (64x64 크기로)
def load_bot_image(bot_name):
    global addon_image_paths
    # addon_image_paths에 봇 이미지 경로가 등록되어 있으면 해당 폴더 사용, 아니면 기본 assets 폴더 사용
    if bot_name in addon_image_paths:
        assets_dir = addon_image_paths[bot_name]
    else:
        assets_dir = os.path.join(os.getcwd(), "assets")
    
    image_path = os.path.join(assets_dir, f"{bot_name}.png")
    if not os.path.exists(image_path):
        # addon 폴더 내에 default.png가 있다면 사용, 없으면 os.getcwd()의 default.png 사용
        alt_path = os.path.join(assets_dir, "default.png")
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            image_path = os.path.join(os.getcwd(), "default.png")
    
    pixmap = QPixmap(image_path)
    if pixmap.isNull():
        print(f"Failed to load image: {image_path}")
        pixmap = QPixmap(os.path.join(os.getcwd(), "default.png"))
    return pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

@bot_decorator("tit-for-tat Bot", "a simple bot that trusts on the first move and then mimics the opponent.")
def t4t(input):
    if round == 0:
        return "trust"
    if input == 1:
        return b2log[-1]
    else:
        return b1log[-1]

@bot_decorator("t4t variation: trust again", "a simple bot that trusts on the first move and then mimics the opponent. But it has a 10 percent chance to trust the bot again")
def t4t_TA(input):
    if round == 0:
        return "trust"
    if random.randint(1,10) == 1:
        return "trust"
    else:
        if input == 1:
            return b2log[-1]
        else:
            return b1log[-1]

@bot_decorator("t4t variation: 2 tat", "a simple bot that trusts when the opponent doesn't betray twice in a row")
def t42t(input):
    # tit-for-2-tat: 첫 판에 신뢰, 상대가 2번 연속 배신할 때까지 신뢰
    if round <= 1:
        return "trust"
    if input == 1:
        if b2log[-1] == "betray" and b2log[-2] == "betray":
            return "betray"
        else:
            return "trust"
    else:
        if b1log[-1] == "betray" and b1log[-2] == "betray":
            return "betray"
        else:
            return "trust"

@bot_decorator("t4t variation: greedy", "a simple bot that trusts on the first move and then mimics the opponent. But it has a 10 percent chance to trust the bot again")
def greedy_t4t(input):
    # 욕심 많은 tit-for-tat: 첫 판 신뢰, 이후 1/10 확률 배신, 아니면 상대 마지막 행동.
    if round == 0:
        return "trust"
    if random.randint(1,10) == 1:
        return "betray"
    else:
        if input == 1:
            return b2log[-1]   # 수정
        else:
            return b1log[-1]   # 수정


def B_t4t(input):
    # bad tit-for-tat: 첫 판 배신, 이후 상대의 마지막 행동 따라감.
    if round == 0:
        return "betray"
    if input == 1:
        return b2log[-1]   # 수정
    else:
        return b1log[-1]   # 수정

def betrayer(input):
    return "betray"

def truster(input):
    return "trust"

def joker(input):
    return "trust" if random.randint(0, 1) == 0 else "betray"

def gruder(input):
    # 상대가 한 번이라도 배신하면 계속 배신 (첫 판 신뢰)
    if round == 0:
        return "trust"
    if input == 1:
        if "betray" in b2log:
            return "betray"
        else:
            return "trust"
    else:
        if "betray" in b1log:
            return "betray"
        else:
            return "trust"

def researcher(input):
    # 상대가 가장 많이 하는 행동에 따라 결정 (첫 판 신뢰)
    if round == 0:
        return "trust"
    else:
        if input == 1:
            if morecommon(b2log) == "trust":
                return "trust"
            else:
                return "betray"
        else:
            if morecommon(b1log) == "trust":
                return "trust"
            else:
                return "betray"

# tester 함수는 이미 global 선언이 되어있으므로 그대로 사용
def tester(input):
    global tester_retribution, tester_decision
    if round == 0:
        return "betray"
    if round == 1:
        return "trust"
    if 2 <= round <= 20:
        if input == 1:
            if b1log[-2] == "betray" and b2log[-1] == "betray":
                tester_retribution += 1
        else:
            if b2log[-2] == "betray" and b1log[-1] == "betray":
                tester_retribution += 1
        return "trust" if random.randint(0, 1) == 0 else "betray"
    if round == 21:
        if tester_retribution >= 6:
            tester_decision = "trust"
        else:
            tester_decision = "betray"
    if round >= 21:
        return tester_decision

def transquilizer(whichbotitits):#20번 신뢰 후 1/5 배신
    if round <= 20:
        return "trust"
    else:
        return "betray" if random.randint(1, 5) == 1 else "trust"

def adaptive_tit_for_tat(input):
    """상대가 배신하면 2라운드 동안 보복 후 신뢰로 복귀"""
    global round, b1log, b2log
    if round == 0:
        adaptive_tit_for_tat.revenge_counter = 0
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    if opp == "betray":
        adaptive_tit_for_tat.revenge_counter = 2
    if getattr(adaptive_tit_for_tat, "revenge_counter", 0) > 0:
        adaptive_tit_for_tat.revenge_counter -= 1
        return "betray"
    return "trust"

def forgiving_tit_for_tat(input):
    """배신 시 25% 확률로 용서 후 신뢰, 아니면 보복"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    if opp == "betray":
        return "trust" if random.random() < 0.25 else "betray"
    return "trust"

def reverse_tit_for_tat(input):
    """상대가 신뢰하면 배신, 배신하면 신뢰 (심리전)"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    return "betray" if opp == "trust" else "trust"

def randomized_tit_for_tat(input):
    """기본적으로 상대 행동 모방, 10% 확률로 무작위 선택"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    if random.random() < 0.1:
        return "trust" if random.random() < 0.5 else "betray"
    return b2log[-1] if input == 1 else b1log[-1]

def soft_tit_for_tat(input):
    """상대 배신 시 50% 확률로 보복, 50% 신뢰"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    if opp == "betray":
        return "betray" if random.random() < 0.5 else "trust"
    return "trust"

def t4t_3tb(input):
    """상대가 3회 연속 협력하면 배신 기회 포착"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp_log = b2log if input == 1 else b1log
    if len(opp_log) >= 3 and opp_log[-1] == "trust" and opp_log[-2] == "trust" and opp_log[-3] == "trust":
        return "betray"
    return opp_log[-1]

def mirror_bot(input):
    """상대 전체 행동 빈도를 반영해 결정"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp_log = b2log if input == 1 else b1log
    return "trust" if opp_log.count("trust") >= opp_log.count("betray") else "betray"

def soft_betrayer(input):
    """80% 확률로 배신, 20% 신뢰"""
    return "betray" if random.random() < 0.8 else "trust"

def soft_truster(input):
    """80% 확률로 신뢰, 20% 배신"""
    return "trust" if random.random() < 0.8 else "betray"

def pacifist(input):
    """첫 10판 무조건 신뢰, 이후 Tit-for-Tat 적용"""
    global round, b1log, b2log
    if round < 10:
        return "trust"
    return b2log[-1] if input == 1 else b1log[-1]

def overly_optimistic_bot(input):
    """상대가 한 번 신뢰하면 계속 신뢰, 없으면 배신"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp_log = b2log if input == 1 else b1log
    return "trust" if "trust" in opp_log else "betray"

def countdown_betrayer(input):
    """정해진 라운드(예: 20판)까지 신뢰, 이후 계속 배신"""
    global round
    return "trust" if round < 20 else "betray"

def minority_rule(input):
    """전체 기록에서 상대 소수 행동에 따라 결정"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp_log = b2log if input == 1 else b1log
    return "trust" if opp_log.count("trust") < opp_log.count("betray") else "betray"

def memory_bot(input):
    """최근 10판의 상대 기록에 따라 결정"""
    global round, b1log, b2log
    opp = (b2log if input == 1 else b1log)[-10:]
    return "trust" if opp.count("trust") >= opp.count("betray") else "betray"

def learning_bot(input):
    """상대 패턴을 학습해 신뢰 확률 조정 (간단 가중평균)"""
    global round, b1log, b2log
    if round == 0:
        learning_bot.prob = 0.5
        return "trust"
    opp_log = b2log if input == 1 else b1log
    trust_rate = opp_log.count("trust") / len(opp_log)
    learning_bot.prob = (learning_bot.prob + trust_rate) / 2
    return "trust" if random.random() < learning_bot.prob else "betray"

def predictive_bot(input):
    """상대의 마지막 행동 예측 (실질적 Tit-for-Tat)"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp_log = b2log if input == 1 else b1log
    return opp_log[-1]

def noise_resistant_tit_for_tat(input):
    """배신 시 30% 확률로 용서하는 Tit-for-Tat"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    if opp == "betray" and random.random() < 0.3:
        return "trust"
    return opp

def fairness_bot(input):
    """자신과 상대 점수 차에 따라 결정 (점수 차가 크면 배신)"""
    global b1point, b2point
    diff = (b1point - b2point) if input == 1 else (b2point - b1point)
    return "betray" if diff > 5 else "trust"

def switching_strategy(input):
    """상대 배신이 많으면 즉시 항상 배신 모드 전환"""
    global round, b1log, b2log
    if round == 0:
        switching_strategy.switched = False
        return "trust"
    opp_log = b2log if input == 1 else b1log
    if opp_log.count("betray") > 3:
        switching_strategy.switched = True
    return "betray" if switching_strategy.switched else (opp_log[-1] if opp_log else "trust")

def evolving_strategy(input):
    """초반엔 신뢰, 최근 5판 배신 2회 이상이면 배신, 아니면 Tit-for-Tat"""
    global round, b1log, b2log
    if round < 5:
        return "trust"
    opp_log = b2log if input == 1 else b1log
    if opp_log[-5:].count("betray") > 2:
        return "betray"
    return opp_log[-1]

def decaying_trust(input):
    """라운드가 지날수록 신뢰 확률 선형 감소"""
    global round
    p_trust = max(0, 1 - round * 0.01)
    return "trust" if random.random() < p_trust else "betray"

def trend_follower(input):
    """최근 3~5판 추세에 따라 결정 (협력이 다수면 협력)"""
    global round, b1log, b2log
    if round < 3:
        return "trust"
    opp_log = b2log if input == 1 else b1log
    return "trust" if opp_log[-3:].count("trust") >= 2 else "betray"

def revenge_bot(input):
    """최근 3판 중 배신 2회 이상이면 보복"""
    global round, b1log, b2log
    if round < 3:
        return "trust"
    opp_log = b2log if input == 1 else b1log
    return "betray" if opp_log[-3:].count("betray") >= 2 else "trust"

def friendly_tit_for_tat(input):
    """상대 마지막 행동 모방 + 신뢰 시 추가 신뢰 확률 부여"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    if opp == "trust" and random.random() < 0.2:
        return "trust"
    return opp

def decisive_bot(input):
    """상대가 장기간 배신하면 단 한 번 강한 보복 후 원상복귀"""
    global round, b1log, b2log
    if round < 5:
        return "trust"
    opp_log = b2log if input == 1 else b1log
    if opp_log.count("betray") >= 3:
        return "betray"
    return "trust"

def trap_bot(input):
    """초반 신뢰 후 5라운드에서 배신, 그 후 Tit-for-Tat 적용"""
    global round, b1log, b2log
    if round < 5:
        return "trust"
    if round == 5:
        return "betray"
    return b2log[-1] if input == 1 else b1log[-1]

def diplomatic_bot(input):
    """배신 시 한 번 용서하는 외교적 전략"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    opp_log = b2log if input == 1 else b1log
    if opp == "betray" and opp_log[-3:].count("betray") == 1:
        return "trust"
    return opp

def calculated_gambler(input):
    """단순 계산: 짝수 라운드에 배신, 홀수에 신뢰"""
    global round
    return "betray" if round % 2 == 0 else "trust"

def meta_learner(input):
    """여러 전략(티트포탯, 다수결, 홀짝 등)을 혼합해 결정"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    strat1 = b2log[-1] if input == 1 else b1log[-1]
    opp_log = b2log if input == 1 else b1log
    strat2 = "trust" if opp_log.count("trust") >= opp_log.count("betray") else "betray"
    strat3 = "betray" if round % 2 == 0 else "trust"
    choices = [strat1, strat2, strat3]
    return "trust" if choices.count("trust") > choices.count("betray") else "betray"

def stubborn_tit_for_tat(input):
    """상대 배신 후 3라운드 뒤에야 반응"""
    global round, b1log, b2log
    if round < 3:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    if (b2log if input==1 else b1log)[-3] == "betray":
        return "betray"
    return "trust"

def opportunist(input):
    """초반 협력 후, 상대 배신 시 간헐적 보복"""
    global round, b1log, b2log
    if round < 5:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    return "betray" if (opp == "betray" and round % 2 == 0) else "trust"

def risk_averse_bot(input):
    """명확한 배신 신호 있을 때만 배신"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    return "betray" if opp == "betray" else "trust"

def adaptive_oscillator(input):
    """점수 차에 따라 교대로 행동 (리드 중이면 공격적)"""
    global round, b1point, b2point
    diff = (b1point - b2point) if input == 1 else (b2point - b1point)
    return "betray" if diff > 0 and round % 2 == 0 else "trust"

def conditional_mirror(input):
    """상대 신뢰 비율이 50% 이상이면 모방, 아니면 배신"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp = b2log if input == 1 else b1log
    if opp.count("trust") >= len(opp) / 2:
        return opp[-1]
    else:
        return "betray"

def weighted_memory_bot(input):
    """최근 5~3판에 가중치 부여 후 결정"""
    global round, b1log, b2log
    opp = b2log if input == 1 else b1log
    weighted_trust = sum(2 if idx >= len(opp)-3 and move=="trust" else 1 if move=="trust" else 0 for idx, move in enumerate(opp))
    weighted_betray = sum(2 if idx >= len(opp)-3 and move=="betray" else 1 if move=="betray" else 0 for idx, move in enumerate(opp))
    return "trust" if weighted_trust >= weighted_betray else "betray"

def threshold_bot(input):
    """상대 배신 빈도가 3회 초과면 배신"""
    global round, b1log, b2log
    if round < 5:
        return "trust"
    opp = b2log if input == 1 else b1log
    return "betray" if opp.count("betray") > 3 else "trust"

def dynamic_threshold_bot(input):
    """라운드 진행에 따라 임계치 조정 후 결정"""
    global round, b1log, b2log
    threshold = 3 + round // 10
    opp = b2log if input == 1 else b1log
    return "betray" if opp.count("betray") >= threshold else "trust"

def greedy_opportunist(input):
    """자신 점수가 낮으면 공격적으로 배신"""
    global round, b1point, b2point, b1log, b2log
    diff = (b1point - b2point) if input == 1 else (b2point - b1point)
    if diff < 0:
        return "betray"
    if round >= 2:
        return b2log[-1] if input == 1 else b1log[-1]
    else:
        return "betray"

def scorekeeper_bot(input):
    """자신의 누적 점수에 따라 결정 (낮은 점수면 협력)"""
    global b1point, b2point
    if input == 1:
        return "trust" if b1point <= b2point else "betray"
    else:
        return "trust" if b2point <= b1point else "betray"

def early_betrayer(input):
    """초반 3판에서 배신, 이후 신뢰"""
    global round
    if round < 3:
        return "betray"
    return "trust"

def adaptive_early_betrayer(input):
    """초반 배신 후, 이후엔 Tit-for-Tat 전환"""
    global round, b1log, b2log
    if round < 3:
        return "betray"
    return b2log[-1] if input == 1 else b1log[-1]

def exponential_betrayer(input):
    """라운드 진행에 따라 기하급수적 배신 확률 증가"""
    global round
    p = 1 - math.exp(-0.05 * round)
    return "betray" if random.random() < p else "trust"

def linear_betrayer(input):
    """라운드 진행에 따라 선형적으로 배신 확률 증가"""
    global round
    p = min(1, round * 0.01)
    return "betray" if random.random() < p else "trust"

def probabilistic_mirror(input):
    """상대 마지막 행동 모방하되, 확률적으로 약간 변동"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    if opp == "trust":
        return "trust" if random.random() < 0.8 else "betray"
    else:
        return "betray" if random.random() < 0.8 else "trust"

def decremental_bot(input):
    """상대 협력이 지속되면 배신 확률 서서히 낮춤"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log
    p = 0.5 - (opp.count("trust") / len(opp) * 0.3) if len(opp) > 0 else 0.5
    return "betray" if random.random() < p else "trust"

def frequency_tracker(input):
    """상대 행동 빈도의 역수를 적용해 결정"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log
    return "betray" if opp.count("trust") > opp.count("betray") else "trust"

def multi_phase_bot(input):
    """여러 단계(협력→중립→배신)로 전환"""
    global round
    if round < 10:
        return "trust"
    elif round < 20:
        return "betray"
    else:
        return "trust"

def adaptive_multi_phase_bot(input):
    """상대 반응에 따라 단계 전환 시점 조정"""
    global round, b1log, b2log
    if round < 10:
        return "trust"
    elif round < 20:
        opp = b2log if input == 1 else b1log
        return "betray" if opp.count("betray") > opp.count("trust") else "trust"
    else:
        return b2log[-1] if input == 1 else b1log[-1]

def calculative_bot(input):
    """간단 계산: 초반 10판 협력, 이후 배신"""
    global round
    return "trust" if round < 10 else "betray"

def cautious_counter_bot(input):
    """마지막 행동이 배신이면 배신, 아니면 70% 신뢰"""
    global b1log, b2log
    if not (b2log if input == 1 else b1log):
        return "trust"
    last = (b2log if input == 1 else b1log)[-1]
    if last == "betray":
        return "betray"
    return "trust" if random.random() < 0.7 else "betray"

def random_delay_bot(input):
    """초반 5판은 신뢰, 이후 무작위 선택"""
    global round
    if round < 5:
        return "trust"
    return "trust" if random.random() < 0.5 else "betray"

def adaptive_pattern_shifter(input):
    """10라운드마다 전략 전환 (내부 플래그 토글)"""
    global round
    if round % 10 == 0:
        adaptive_pattern_shifter.shift = not getattr(adaptive_pattern_shifter, 'shift', False)
    return "betray" if getattr(adaptive_pattern_shifter, 'shift', False) else "trust"

def weighted_randomizer(input):
    """상대 행동 가중치에 따라 무작위 결정"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log
    weight = opp.count("trust") - opp.count("betray")
    p = 0.5 + 0.05 * weight
    p = max(0, min(1, p))
    return "trust" if random.random() < p else "betray"

def pressure_bot(input):
    """상대가 지나치게 협력하면 배신 압박"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log
    return "betray" if opp.count("trust") > opp.count("betray") + 2 else "trust"

def escalation_bot(input):
    """배신 감지 시 보복 강도 점진 증가 (level ≥ 2이면 배신)"""
    global round, b1log, b2log
    if not hasattr(escalation_bot, 'level'):
        escalation_bot.level = 0
    if round < 3:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    if opp == "betray":
        escalation_bot.level += 1
    else:
        escalation_bot.level = max(0, escalation_bot.level - 1)
    return "betray" if escalation_bot.level >= 2 else "trust"

def deescalation_bot(input):
    """협력 시 보복 강도 서서히 감소, 배신 시 증가"""
    global round, b1log, b2log
    if not hasattr(deescalation_bot, 'level'):
        deescalation_bot.level = 2
    if round < 3:
        return "trust"
    opp = b2log[-1] if input == 1 else b1log[-1]
    if opp == "trust":
        deescalation_bot.level = max(0, deescalation_bot.level - 1)
    else:
        deescalation_bot.level += 1
    return "betray" if deescalation_bot.level >= 2 else "trust"

def probabilistic_adaptive_bot(input):
    """상대 신뢰율에 따라 확률적으로 결정"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log
    p = opp.count("trust") / len(opp) if len(opp) > 0 else 0.5
    return "trust" if random.random() < p else "betray"

def consistency_checker(input):
    """최근 3회 행동이 모두 동일하면 그대로 따름"""
    global b1log, b2log, round
    if round < 3:
        return "trust"
    opp = b2log if input == 1 else b1log
    if opp[-3:] == ["trust", "trust", "trust"]:
        return "trust"
    if opp[-3:] == ["betray", "betray", "betray"]:
        return "betray"
    return "trust"

def reverse_consistency_checker(input):
    """상대가 너무 일관되면 의도적으로 반대로 결정"""
    global b1log, b2log, round
    if round < 3:
        return "betray"
    opp = b2log if input == 1 else b1log
    if opp[-3:] == ["trust", "trust", "trust"]:
        return "betray"
    if opp[-3:] == ["betray", "betray", "betray"]:
        return "trust"
    return "trust"

def adaptive_hybrid_bot(input):
    """Tit-for-Tat와 다수결 전략 혼합"""
    global b1log, b2log, round
    if round == 0:
        return "trust"
    opp = b2log if input == 1 else b1log
    decision1 = opp[-1]
    decision2 = "trust" if opp.count("trust") >= opp.count("betray") else "betray"
    return decision1 if decision1 == decision2 else "trust"

def statistical_outlier_bot(input):
    """상대 마지막 행동이 전체 평균에서 크게 벗어나면 보복"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log
    if len(opp) == 0:
        return "trust"
    avg = opp.count("trust") / len(opp)
    last = 1 if opp[-1] == "trust" else 0
    return "betray" if abs(last - avg) > 0.5 else "trust"

def cycle_detector(input):
    """최근 6회 기록에서 3회 주기 패턴 발견 시 배신"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log
    if len(opp) < 6:
        return "trust"
    if opp[-4:-2] == opp[-6:-4]:
        return "betray"
    return "trust"

def adaptive_window_bot(input):
    """기억 창 크기를 라운드에 따라 확장해 결정"""
    global b1log, b2log, round
    window = 5 + round // 10
    opp = (b2log if input == 1 else b1log)[-window:]
    return "trust" if opp.count("trust") >= opp.count("betray") else "betray"

def counter_adaptive_bot(input):
    """상대의 적응 경향과 반대로 결정"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log
    decision = "trust" if opp.count("trust") >= opp.count("betray") else "betray"
    return "betray" if decision == "trust" else "trust"

def secret_signal_bot(input):
    """상대의 마지막 3회가 모두 협력하면 신호 보내 배신"""
    global b1log, b2log, round
    if round < 3:
        return "trust"
    opp = b2log if input == 1 else b1log
    secret_signal_bot.signal = (opp[-3:] == ["trust", "trust", "trust"])
    return "betray" if secret_signal_bot.signal else "trust"

def blue_flag_bot(input):
    """최근 5회 중 3회 이상 협력이면 신뢰, 아니면 배신"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log
    return "trust" if len(opp) >= 5 and opp[-5:].count("trust") >= 3 else "betray"

def defensive_gambler(input):
    """상대 배신 시 70% 확률 배신, 협력 시 70% 신뢰"""
    global b1log, b2log
    if round == 0:
        return "trust"
    else:
        opp = b2log[-1] if input == 1 else b1log[-1]
        if opp == "betray":
            return "betray" if random.random() < 0.7 else "trust"
        else:
            return "trust" if random.random() < 0.7 else "betray"

def intermittent_bot(input):
    """7라운드마다 무작위 결정, 그 외엔 신뢰"""
    global round
    if round % 7 == 0:
        return "trust" if random.random() < 0.5 else "betray"
    return "trust"

def self_correcting_bot(input):
    """자신의 점수가 뒤처지면 배신으로 전환"""
    global b1point, b2point
    if input == 1:
        return "betray" if b1point < b2point else "trust"
    else:
        return "betray" if b2point < b1point else "trust"

def adaptive_learning_bot(input):
    """간단한 강화학습 기반 (과거 신뢰율 평균)"""
    global round, b1log, b2log
    if round == 0:
        adaptive_learning_bot.q = 0.5
        return "trust"
    opp = b2log if input == 1 else b1log
    trust_rate = opp.count("trust") / len(opp)
    adaptive_learning_bot.q = (adaptive_learning_bot.q + trust_rate) / 2
    return "trust" if random.random() < adaptive_learning_bot.q else "betray"

def hybrid_strategy_bot(input):
    """여러 전략(항상 신뢰, 항상 배신, Tit-for-Tat)을 무작위 선택"""
    global round, b1log, b2log
    strategies = [
        lambda: b2log[-1] if input == 1 and b2log else "trust",
        lambda: "trust" if (b2log if input == 1 else b1log).count("trust") >= (b2log if input == 1 else b1log).count("betray") else "betray",
        lambda: "betray" if round % 2 == 0 else "trust",
        lambda: "trust" if b1point <= b2point else "betray"
    ]
    return random.choice(strategies)()

def turnaround_bot(input):
    """점수 차에 따라 급격한 전략 전환"""
    global b1point, b2point
    diff = (b1point - b2point) if input == 1 else (b2point - b1point)
    return "betray" if diff < 0 else "trust"

def adaptive_steady_bot(input):
    """상대 행동 비율에 따라 고정 비율 서서히 조정"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log
    ratio = opp.count("trust") / len(opp) if len(opp) > 0 else 0.5
    return "trust" if ratio > 0.5 else "betray"

def temperature_adjusted_tit_for_tat(input):
    """Tit-for-Tat에 온도 변수 적용해 무작위성 조절"""
    global round, b1log, b2log
    if round == 0:
        return "trust"
    temp = min(1, round * 0.01)
    base = b2log[-1] if input == 1 else b1log[-1]
    if random.random() < temp:
        return "betray" if base == "trust" else "trust"
    return base

def convergence_bot(input):
    """시간이 지남에 따라 협력 패턴으로 수렴"""
    global round
    return "trust" if round > 20 else "betray"

def self_tuning_bot(input):
    """게임 진행 중 자신의 매개변수를 실시간 조정"""
    global b1point, b2point
    diff = (b1point - b2point) if input == 1 else (b2point - b1point)
    return "trust" if diff < 0 else "betray"

def reliability_bot(input):
    """상대 행동 일관성(표준편차)에 따라 결정: std < 0.3이면 신뢰"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log
    if len(opp) < 2:
        return "trust"
    encoded = [1 if move == "trust" else 0 for move in opp]
    std = statistics.pstdev(encoded)
    return "trust" if std < 0.3 else "betray"

def adaptive_ratio_bot(input):
    """목표 점수에 따라 신뢰:배신 비율 동적 조정"""
    global round, b1point, b2point
    diff = (b1point - b2point) if input == 1 else (b2point - b1point)
    ratio = 0.5 + diff / 100.0
    ratio = max(0, min(1, ratio))
    return "trust" if random.random() < ratio else "betray"

def ultimate_hybrid_bot(input):
    """Tit-for-Tat, 다수결, 홀짝, 점수 기반 등 여러 전략 복합 적용"""
    global round, b1log, b2log, b1point, b2point
    strategies = [
        lambda: b2log[-1] if input == 1 and b2log else "trust",
        lambda: "trust" if (b2log if input == 1 else b1log).count("trust") >= (b2log if input == 1 else b1log).count("betray") else "betray",
        lambda: "betray" if round % 2 == 0 else "trust",
        lambda: "trust" if b1point <= b2point else "betray"
    ]
    return random.choice(strategies)()
    

def oddbetrayeventrust(input):
    """전체 점수 합이 짝수면 신뢰, 홀수면 배신"""
    global b1point, b2point
    total = b1point + b2point
    return "trust" if total % 2 == 0 else "betray"

def opponentdoublemove(input):
    """상대 마지막 두 행동이 같으면 배신, 아니면 신뢰"""
    global b1log, b2log, round
    opp = b2log if input == 1 else b1log
    if len(opp) < 2:
        return "trust"
    return "betray" if opp[-1] == opp[-2] else "trust"

def eventrusttrust(input):
    """상대 협력 횟수가 짝수면 신뢰, 홀수면 배신"""
    global b1log, b2log
    opp = b2log if input == 1 else b1log 
    return "trust" if opp.count("trust") % 2 == 0 else "betray"


# 초기: 오리지널 봇/애드온 봇 (bot_functions는 나중에 업데이트됨)
original_bots = bot_functions.copy()
addons_bots = {}

###########################################################
# 메인 윈도우 (UI 구조 변경)
###########################################################
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bot Tournament & 1v1 UI (PyQt6)")
        self.resize(1200, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        self.create_top_frame()
        self.create_middle_frame()
        self.create_bottom_panel()

        # 하단 Bot List 갱신용 타이머 생성
        self.bottom_refresh_timer = QTimer(self)
        self.bottom_refresh_timer.setSingleShot(True)
        self.bottom_refresh_timer.timeout.connect(self.refresh_bottom_bot_list)
        self.bottom_scroll_area.viewport().installEventFilter(self)

        global bot_functions, total_bot_count, original_bots
        all_funcs = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        bot_functions = [n for n, f in all_funcs if n not in excluded]
        total_bot_count = len(bot_functions)
        original_bots = bot_functions.copy()
        self.update_bot_notebook()
        self.update_tournament_participants()
        self.refresh_bottom_bot_list()

    def create_top_frame(self):
        top_frame = QFrame()
        top_layout = QHBoxLayout(top_frame)

        self.mode_button_group = QButtonGroup(self)
        self.radio_tournament = QRadioButton("Tournament")
        self.radio_1v1 = QRadioButton("1v1")
        self.mode_button_group.addButton(self.radio_tournament)
        self.mode_button_group.addButton(self.radio_1v1)
        self.radio_tournament.setChecked(True)
        top_layout.addWidget(self.radio_tournament)
        top_layout.addWidget(self.radio_1v1)

        self.addon_button = QPushButton("Load Addons")
        self.addon_button.clicked.connect(self.load_all_addons)
        top_layout.addWidget(self.addon_button)

        self.select_bots_button = QPushButton("Select Bots")
        self.select_bots_button.clicked.connect(self.open_select_bots_window)
        top_layout.addWidget(self.select_bots_button)

        top_layout.addWidget(QLabel("Rounds:"))
        self.rounds_entry = QLineEdit("50")
        self.rounds_entry.setFixedWidth(50)
        top_layout.addWidget(self.rounds_entry)

        top_layout.addWidget(QLabel("1v1 - Bot1 #:"))
        self.bot1_entry = QLineEdit()
        self.bot1_entry.setFixedWidth(50)
        top_layout.addWidget(self.bot1_entry)
        top_layout.addWidget(QLabel("Bot2 #:"))
        self.bot2_entry = QLineEdit()
        self.bot2_entry.setFixedWidth(50)
        top_layout.addWidget(self.bot2_entry)

        self.go_button = QPushButton("Go")
        self.go_button.clicked.connect(self.select_mode)
        top_layout.addWidget(self.go_button)

        self.progress_label = QLabel("Progress: 0%")
        top_layout.addWidget(self.progress_label)

        self.main_layout.addWidget(top_frame)

    def create_middle_frame(self):
        middle_frame = QFrame()
        middle_layout = QHBoxLayout(middle_frame)
        self.bot_notebook = QTabWidget()
        
        # 좌측: 토너먼트 참가 봇 이미지를 표시하는 단일 스크롤 영역
        self.tournament_participants_area = QScrollArea()
        self.tournament_participants_area.setWidgetResizable(True)
        self.tournament_participants_widget = QWidget()
        self.tournament_participants_layout = QGridLayout(self.tournament_participants_widget)
        self.tournament_participants_widget.setLayout(self.tournament_participants_layout)
        self.tournament_participants_area.setWidget(self.tournament_participants_widget)
        middle_layout.addWidget(self.tournament_participants_area, stretch=1)
        
        # 우측: QTabWidget을 생성하여 Graph와 Rankings 탭을 추가
        self.right_tab_widget = QTabWidget()
        
        # Graph 탭
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.fig.patch.set_facecolor("#303030")
        self.ax.set_facecolor("#303030")
        self.ax.set_xlabel("Score", color="white")
        self.ax.set_title("Bot Scores", color="white")
        graph_layout.addWidget(self.canvas)
        graph_tab.setLayout(graph_layout)
        
        # Rankings 탭
        rankings_tab = QWidget()
        rankings_layout = QVBoxLayout(rankings_tab)
        self.ranking_text = QPlainTextEdit()
        self.ranking_text.setReadOnly(True)
        rankings_layout.addWidget(self.ranking_text)
        rankings_tab.setLayout(rankings_layout)
        
        self.right_tab_widget.addTab(graph_tab, "Graph")
        self.right_tab_widget.addTab(rankings_tab, "Rankings")
        
        middle_layout.addWidget(self.right_tab_widget, stretch=2)
        
        middle_frame.setLayout(middle_layout)
        self.main_layout.addWidget(middle_frame, stretch=2)
        
        self.update_tournament_participants()

    def create_bottom_panel(self):
        bottom_panel = QGroupBox("Bot List")
        bottom_layout = QVBoxLayout(bottom_panel)

        # 상단 컨트롤 영역: 카테고리 선택 및 검색
        top_bottom_layout = QHBoxLayout()
        top_bottom_layout.addWidget(QLabel("Select Category:"))
        self.bottom_category_combo = QComboBox()
        self.bottom_category_combo.setMinimumWidth(300)  # 여기서 가로 길이 설정 (예: 200 픽셀)
        self.bottom_category_combo.currentIndexChanged.connect(self.refresh_bottom_bot_list)
        top_bottom_layout.addWidget(self.bottom_category_combo)
        top_bottom_layout.addWidget(QLabel("Search:"))
        self.bottom_search_line = QLineEdit()
        self.bottom_search_line.textChanged.connect(self.refresh_bottom_bot_list)
        top_bottom_layout.addWidget(self.bottom_search_line)
        bottom_layout.addLayout(top_bottom_layout)

        # Bot List를 위한 스크롤 영역 (이미지로 표시)
        self.bottom_scroll_area = QScrollArea()
        self.bottom_scroll_area.setWidgetResizable(True)
        self.bottom_bot_list_container = QWidget()
        self.bottom_bot_list_layout = QGridLayout(self.bottom_bot_list_container)
        self.bottom_bot_list_container.setLayout(self.bottom_bot_list_layout)
        self.bottom_scroll_area.setWidget(self.bottom_bot_list_container)
        bottom_layout.addWidget(self.bottom_scroll_area)

        # 메인 레이아웃에 하단 패널 추가
        self.main_layout.addWidget(bottom_panel, stretch=1)
        self.init_bottom_panel_categories()
        self.refresh_bottom_bot_list()

    def on_tab_select(self):
        selected = self.tab_selector.currentText()
        print("Selected tab:", selected)

    def refresh_bottom_bot_list(self):
        self.clear_layout(self.bottom_bot_list_layout)
        category = self.bottom_category_combo.currentText()
        search_text = self.bottom_search_line.text().strip().lower()
        bots = self.bottom_category_dict.get(category, [])
        filtered_bots = [bot for bot in bots if search_text in bot.lower()]
        image_size = 64
        spacing = 10
        available_width = self.bottom_scroll_area.viewport().width()
        num_columns = max(1, available_width // (image_size + spacing))
        print(f"Bottom list refresh: available_width={available_width}, num_columns={num_columns}")
        for idx, bot in enumerate(filtered_bots):
            row = idx // num_columns
            col = idx % num_columns
            pixmap = load_bot_image(bot)
            label = QLabel(self.bottom_bot_list_container)
            label.setPixmap(pixmap)
            label.setFixedSize(image_size, image_size)
            func = globals().get(bot)
            bot_meta_name = getattr(func, "bot_name", bot) if func is not None else bot
            bot_description = getattr(func, "bot_description", "No description") if func is not None else "No description"
            # textwrap.fill로 50글자마다 스페이스에서 줄바꿈
            wrapped_name = textwrap.fill(bot_meta_name, width=50, break_long_words=False, break_on_hyphens=False)
            wrapped_description = textwrap.fill(bot_description, width=50, break_long_words=False, break_on_hyphens=False)
            label.setToolTip(f"{idx+1}. {wrapped_name}\n{wrapped_description}")
            self.bottom_bot_list_layout.addWidget(label, row, col)

    def update_tournament_participants(self):
        self.clear_layout(self.tournament_participants_layout)
        image_size = 64
        spacing = 10
        available_width = self.tournament_participants_area.viewport().width()
        num_columns = max(1, available_width // (image_size + spacing))
        for idx, bot in enumerate(selected_bots_global):
            row = idx // num_columns
            col = idx % num_columns
            pixmap = load_bot_image(bot)
            label = QLabel(self.tournament_participants_widget)
            label.setPixmap(pixmap)
            label.setFixedSize(image_size, image_size)
            func = globals().get(bot)
            bot_meta_name = getattr(func, "bot_name", bot) if func is not None else bot
            bot_description = getattr(func, "bot_description", "No description") if func is not None else "No description"
            wrapped_name = textwrap.fill(bot_meta_name, width=50, break_long_words=False, break_on_hyphens=False)
            wrapped_description = textwrap.fill(bot_description, width=50, break_long_words=False, break_on_hyphens=False)
            label.setToolTip(f"{idx+1}. {wrapped_name}\n{wrapped_description}")
            self.tournament_participants_layout.addWidget(label, row, col)

    def eventFilter(self, source, event):
        if source == self.bottom_scroll_area.viewport() and event.type() == QEvent.Type.Resize:
            self.bottom_refresh_timer.start(100)
        return super().eventFilter(source, event)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def init_bottom_panel_categories(self):
        global original_bots, addons_bots
        # 동일한 로직으로 카테고리 딕셔너리 초기화
        self.bottom_category_dict = {}
        # 만약 original_bots가 비어있으면 빈 리스트, 아니면 복사
        self.bottom_category_dict["Original"] = original_bots[:] if original_bots else []
        # addons_bots가 존재하면 추가
        if addons_bots:
            for addon_name in sorted(addons_bots.keys()):
                self.bottom_category_dict[addon_name] = addons_bots[addon_name][:]
        self.bottom_category_combo.clear()
        for category in self.bottom_category_dict.keys():
            self.bottom_category_combo.addItem(category)


    def update_tab_selector(self):
        self.tab_selector.blockSignals(True)
        self.tab_selector.clear()
        for i in range(self.bot_notebook.count()):
            self.tab_selector.addItem(self.bot_notebook.tabText(i))
        if self.bot_notebook.count() > 0:
            self.tab_selector.setCurrentIndex(0)
        self.tab_selector.blockSignals(False)

    def update_bot_notebook(self):
        global original_bots, addons_bots
        self.bot_notebook.clear()
        counter = 1  # 봇 번호 카운터

        # Original 탭
        orig_tab = QWidget()
        orig_layout = QVBoxLayout(orig_tab)
        orig_text = QPlainTextEdit()
        orig_text.setReadOnly(True)
        if original_bots:
            for bot in original_bots:
                orig_text.appendPlainText(f"{counter}. {bot}")
                counter += 1
        else:
            orig_text.appendPlainText("No original bots available.")
        orig_layout.addWidget(orig_text)
        self.bot_notebook.addTab(orig_tab, "Original")

        # 애드온 탭들
        if not addons_bots:
            empty_tab = QWidget()
            empty_layout = QVBoxLayout(empty_tab)
            empty_text = QPlainTextEdit()
            empty_text.setReadOnly(True)
            empty_text.appendPlainText("No addon bots available.")
            empty_layout.addWidget(empty_text)
            self.bot_notebook.addTab(empty_tab, "Addons")
        else:
            for addon_name, bot_list in addons_bots.items():
                addon_tab = QWidget()
                addon_layout = QVBoxLayout(addon_tab)
                addon_text = QPlainTextEdit()
                addon_text.setReadOnly(True)
                if bot_list:
                    for bot in bot_list:
                        addon_text.appendPlainText(f"{counter}. {bot}")
                        counter += 1
                else:
                    addon_text.appendPlainText("No bots in this addon.")
                addon_layout.addWidget(addon_text)
                self.bot_notebook.addTab(addon_tab, addon_name)


    def select_mode(self):
        if self.radio_tournament.isChecked():
            self.run_tournament()
        else:
            self.run_1v1()

    def run_tournament(self):
        global overall_scores_global, selected_bots_global
        try:
            rounds_val = int(self.rounds_entry.text())
        except ValueError:
            rounds_val = 50
        if not selected_bots_global:
            QMessageBox.critical(self, "Error", "No bots selected. Please select bots first.")
            return
        overall_scores = {bot: 0 for bot in selected_bots_global}
        battle_pairs = list(itertools.combinations(selected_bots_global, 2))
        total_battles = len(battle_pairs)
        battle_count = 0
        for bot1, bot2 in battle_pairs:
            scores = battle(bot1, bot2, rounds_val, print_result=False)
            overall_scores[bot1] += scores.get(bot1, 0)
            overall_scores[bot2] += scores.get(bot2, 0)
            battle_count += 1
            completion_rate = battle_count / total_battles * 100
            self.progress_label.setText(f"Progress: {completion_rate:.2f}%")
            QApplication.processEvents()
        overall_scores_global = overall_scores.copy()
        sorted_scores = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 그래프 업데이트
        self.update_bar_graph(sorted_scores)
        
        # Rankings 탭 업데이트: 순위표 텍스트 생성
        ranking_str = "Tournament Participants Rankings:\n"
        for idx, (bot, score) in enumerate(sorted_scores, start=1):
            ranking_str += f"{idx}. {bot}: {score}\n"
        self.ranking_text.setPlainText(ranking_str)
        
        QMessageBox.information(self, "Tournament", "Tournament complete.")

    def run_1v1(self):
        combinedBots = self.getCombinedBotList()
        try:
            choice1 = int(self.bot1_entry.text()) - 1
            choice2 = int(self.bot2_entry.text()) - 1
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid bot selection.")
            return
        if choice1 < 0 or choice1 >= len(combinedBots) or choice2 < 0 or choice2 >= len(combinedBots):
            QMessageBox.critical(self, "Error", "Invalid bot selection.")
            return
        bot1 = combinedBots[choice1]
        bot2 = combinedBots[choice2]
        try:
            rounds_val = int(self.rounds_entry.text())
        except:
            rounds_val = 50
        scores = battle(bot1, bot2, rounds_val)
        result_text = f"1v1: {bot1} vs {bot2}\nFinal Score: {bot1}({scores.get(bot1, 0)}) vs {bot2}({scores.get(bot2, 0)})"
        QMessageBox.information(self, "1v1 Result", result_text)
        self.update_bar_graph([(bot1, scores.get(bot1, 0)), (bot2, scores.get(bot2, 0))])

    def update_bar_graph(self, scores):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor("#303030")
        ax.set_facecolor("#303030")
        ax.set_xlabel("Score", color="white")
        ax.set_title("Bot Scores", color="white")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        if not scores:
            self.canvas.draw()
            return
        bots = [bot for bot, score in scores]
        scores_list = [score for bot, score in scores]
        bars = ax.barh(range(len(bots)), scores_list, color="skyblue")
        ax.set_yticks(range(len(bots)))
        ax.set_yticklabels([""] * len(bots))
        cursor = mplcursors.cursor(bars, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            idx = sel.index
            sel.annotation.set_text(bots[idx])
        self.canvas.draw()

    def getCombinedBotList(self):
        combined = []
        global original_bots, addons_bots
        combined.extend(original_bots)
        for addon in sorted(addons_bots.keys()):
            combined.extend(addons_bots[addon])
        return combined

    def load_all_addons(self):
        """
        현재 작업 디렉토리의 'addons' 폴더 내에 있는 모든 애드온 폴더를 스캔하여,
        각 폴더에 addon.py가 존재하면 load_addon()을 호출하여 애드온을 로드합니다.
        """
        addons_dir = os.path.join(os.getcwd(), "addons")
        if not os.path.isdir(addons_dir):
            print("No addons folder found.")
            return
        for item in os.listdir(addons_dir):
            addon_path = os.path.join(addons_dir, item)
            if os.path.isdir(addon_path):
                addon_py = os.path.join(addon_path, "addon.py")
                if os.path.exists(addon_py):
                    try:
                        self.load_addon(addon_path)
                        QMessageBox.information(self, "Info", "addons loaded.")
                        print(f"Loaded addon: {item}")
                    except Exception as e:
                        print(f"Failed to load addon {item}: {e}")

    def load_addon(self, addon_folder):
        global addons_bots, addon_image_paths
        import os, sys, ast, subprocess, importlib.util

        addon_basename = os.path.basename(addon_folder)
        addon_py = os.path.join(addon_folder, "addon.py")
        
        # ----------------------------
        # 1. addon.py 내 import 문 분석
        # ----------------------------
        def get_imported_modules(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
            modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # 최상위 모듈명만 취함 (예: numpy.linalg -> numpy)
                        modules.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        modules.add(node.module.split('.')[0])
            return modules

        imported_modules = get_imported_modules(addon_py)
        
        # ----------------------------
        # 2. 각 모듈이 설치되었는지 확인 후, 없으면 설치
        # ----------------------------
        for module in imported_modules:
            # 내장 모듈은 건너뛰기
            if module in sys.builtin_module_names:
                continue
            # 모듈이 발견되지 않으면 설치 시도
            if importlib.util.find_spec(module) is None:
                print(f"Module '{module}' not found. Installing...")
                subprocess.run([sys.executable, "-m", "pip", "install", module])
        
        # ----------------------------
        # 3. addon.py 불러오기 (기존 코드)
        # ----------------------------
        spec = importlib.util.spec_from_file_location("addon_module", addon_py)
        addon_module = importlib.util.module_from_spec(spec)
        
        # 임시로 QWidget.show를 무력화하여 addon 모듈 내 위젯이 보이지 않도록 함.
        from PyQt6.QtWidgets import QWidget
        original_show = QWidget.show
        QWidget.show = lambda self: None

        try:
            spec.loader.exec_module(addon_module)
        finally:
            QWidget.show = original_show

        # 생성된 위젯이 있다면 강제로 숨김.
        for key, obj in addon_module.__dict__.items():
            try:
                if isinstance(obj, QWidget) and obj.isVisible():
                    obj.hide()
            except Exception:
                pass

        bots_list = getattr(addon_module, "bots", [])
        exceptions_list = getattr(addon_module, "exceptions", [])
        excluded.update(exceptions_list)
        
        def get_non_conflicting_name(orig_name):
            new_name = orig_name
            if new_name in globals():
                new_name = f"{addon_basename}_{orig_name}"
            counter = 1
            while new_name in globals():
                new_name = f"{addon_basename}_{orig_name}_{counter}"
                counter += 1
            return new_name

        # 이미 존재하는 변수들은 예외 처리
        exclude_vars = {"b1log", "b2log", "b1play", "b2play", "b1point", "b2point", "round", "bot_decorator"}

        for name, obj in addon_module.__dict__.items():
            if name.startswith("__") or name in exclude_vars:
                continue
            final_name = name if name not in globals() else get_non_conflicting_name(name)
            globals()[final_name] = obj
            if callable(obj) and name in bots_list:
                if addon_basename not in addons_bots:
                    addons_bots[addon_basename] = []
                addons_bots[addon_basename].append(final_name)
                # 이미지 로드 시 사용할 경로는 addon_folder
                addon_image_paths[final_name] = addon_folder

        print(f"Addon '{addon_folder}' loaded successfully!")
        self.init_bottom_panel_categories()


    def open_select_bots_window(self):
        dialog = SelectBotsDialog(self)
        dialog.exec()

class SelectBotsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Bots for Tournament")
        self.resize(600, 500)
        self.layout_main = QVBoxLayout(self)
        
        # 상단: 검색창
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_line = QLineEdit()
        self.search_line.textChanged.connect(self.refresh_checkbox_list)
        search_layout.addWidget(self.search_line)
        self.layout_main.addLayout(search_layout)
        
        # 중앙: 스크롤 영역 (체크박스 목록)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.checkboxes_container = QWidget()
        self.checkboxes_layout = QGridLayout(self.checkboxes_container)
        self.checkboxes_container.setLayout(self.checkboxes_layout)
        self.scroll_area.setWidget(self.checkboxes_container)
        self.layout_main.addWidget(self.scroll_area, stretch=1)
        
        # 하단: 버튼 영역
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        self.btn_select_all = QPushButton("Select All")
        self.btn_select_all.clicked.connect(self.select_all)
        button_layout.addWidget(self.btn_select_all)
        self.btn_unselect_all = QPushButton("Unselect All")
        self.btn_unselect_all.clicked.connect(self.unselect_all)
        button_layout.addWidget(self.btn_unselect_all)
        self.btn_confirm = QPushButton("Confirm Selection")
        self.btn_confirm.clicked.connect(self.confirm_selection)
        button_layout.addWidget(self.btn_confirm)
        self.layout_main.addWidget(button_frame)
        self.setLayout(self.layout_main)
        
        self.checkbox_dict = {}
        # 통합된 봇 목록: 전역 original_bots와 addons_bots를 합침
        global original_bots, addons_bots, selected_bots_global
        self.all_bots = original_bots[:]
        for addon in addons_bots.values():
            self.all_bots.extend(addon)
        # 이전에 선택했던 봇들을 전역 변수에서 불러와 저장
        self.saved_selections = set(selected_bots_global) if selected_bots_global else set()
        self.refresh_checkbox_list()

    def refresh_checkbox_list(self):
        # 먼저 현재 체크박스들의 상태를 저장
        for bot, chk in self.checkbox_dict.items():
            # Qt.CheckState.Checked는 PyQt6의 열거형 값; stateChanged signal 전달값은 int
            if chk.isChecked():
                self.saved_selections.add(bot)
            else:
                self.saved_selections.discard(bot)
                
        self.clear_layout(self.checkboxes_layout)
        search_text = self.search_line.text().strip().lower()
        # 검색어가 포함된 봇들만 필터링
        filtered_bots = [bot for bot in self.all_bots if search_text in bot.lower()]
        self.checkbox_dict = {}
        num_columns = 2  # 한 행에 2개씩 배치 (필요 시 조정)
        for idx, bot in enumerate(filtered_bots):
            row = idx // num_columns
            col = idx % num_columns
            container = QHBoxLayout()
            # 봇 이미지
            pixmap = load_bot_image(bot)
            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setFixedSize(64, 64)
            container.addWidget(img_label)
            # 체크박스: 봇 이름을 표시
            chk = QCheckBox(bot)
            # 초기 상태: 저장된 선택 상태가 있으면 체크
            if bot in self.saved_selections:
                chk.setChecked(True)
            # 연결: 체크박스 상태가 변경될 때마다 update_saved_selection 호출
            chk.stateChanged.connect(lambda state, b=bot: self.update_saved_selection(b, state))
            container.addWidget(chk)
            wrapper = QWidget()
            wrapper.setLayout(container)
            self.checkboxes_layout.addWidget(wrapper, row, col)
            self.checkbox_dict[bot] = chk

    def update_saved_selection(self, bot, state):
        # state는 int: Qt.CheckState.Checked == 2, Qt.CheckState.Unchecked == 0
        if state == Qt.CheckState.Checked.value:
            self.saved_selections.add(bot)
        else:
            self.saved_selections.discard(bot)

    def select_all(self):
        for chk in self.checkbox_dict.values():
            chk.setChecked(True)
        self.saved_selections.update(self.checkbox_dict.keys())

    def unselect_all(self):
        for chk in self.checkbox_dict.values():
            chk.setChecked(False)
        for bot in self.checkbox_dict.keys():
            self.saved_selections.discard(bot)

    def confirm_selection(self):
        global selected_bots_global
        selected_bots_global = list(self.saved_selections)
        self.accept()
        if self.parent() is not None and hasattr(self.parent(), "update_tournament_participants"):
            self.parent().update_tournament_participants()

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()



def main():
    custom_theme_file = os.path.join(os.path.dirname(__file__), "themes", "botbattletheme.xml")
    app = QApplication(sys.argv)
    global bot_functions, total_bot_count, original_bots, main_window
    all_funcs = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    bot_functions = [n for n, f in all_funcs if n not in excluded]
    total_bot_count = len(bot_functions)
    original_bots = bot_functions.copy()
    main_window = MainWindow()
    apply_stylesheet(app, theme=custom_theme_file)
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()