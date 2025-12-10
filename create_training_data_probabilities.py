import numpy as np
import pandas as pd
import time
import random
import os

# 설정
DECK_COUNT = 6
SIMULATION_COUNT = 100000 
OUTPUT_FILE = 'blackjack_probabilities_data.csv'
BATCH_SIZE = 10  # 배치 크기: N개 시나리오마다 파일에 저장

# 카드 정의 (사용자 요청 순서: A ~ K)
SUITS = ['H', 'S', 'D', 'C'] # Heart, Spade, Diamond, Clover
RANKS_DISPLAY = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

# 전체 52장 카드 ID 리스트 생성 (예: 'HA', 'H2', ... 'CK')
ALL_CARD_IDS = [f"{suit}{rank}" for suit in SUITS for rank in RANKS_DISPLAY]
CARD_TO_IDX = {c: i for i, c in enumerate(ALL_CARD_IDS)}

# 값 계산용 매핑
def get_card_value(rank_str):
    if rank_str in ['J', 'Q', 'K', '10']: return 10
    if rank_str == 'A': return 11
    return int(rank_str)

# 랭크별 값 매핑 (확률 계산 엔진용)
# A는 11로 초기화
RANK_TO_VAL_INT = {r: get_card_value(r) for r in RANKS_DISPLAY}

def create_full_deck_values_array():
    """확률 계산용 값(Value) 기반 덱 생성"""
    # 2~9, A, 10(10,J,Q,K) 값들의 분포 생성
    deck = []
    # 2~9, 11(A)는 각각 4(Suit) * 6(Deck) = 24장
    for val in [2,3,4,5,6,7,8,9,11]:
        deck.extend([val] * 24)
    # 10은 16(10,J,Q,K) * 6(Deck) = 96장
    deck.extend([10] * 96)
    return np.array(deck, dtype=np.int8)

def get_remaining_deck_values(full_deck_vals, used_card_ids):
    """
    구체적인 카드 ID(예: 'H2', 'SK') 목록을 받아서,
    확률 계산용 값(Value) 덱에서 해당 카드들의 값을 제거하고 반환
    """
    current_deck = full_deck_vals.copy()
    
    for card_id in used_card_ids:
        # 카드 ID에서 랭크 추출 (예: 'H10' -> '10', 'HA' -> 'A')
        rank = card_id[1:]
        
        # 유효하지 않은 랭크 체크
        if rank not in RANK_TO_VAL_INT:
            raise ValueError(f"Invalid rank '{rank}' extracted from card_id '{card_id}'. Valid ranks: {list(RANK_TO_VAL_INT.keys())}")
        
        val = RANK_TO_VAL_INT[rank]
        
        # 해당 값을 가진 인덱스 중 하나를 제거
        indices = np.where(current_deck == val)[0]
        if len(indices) > 0:
            current_deck = np.delete(current_deck, indices[0])
            
    return current_deck

def calc_bust_results(player_total, is_soft, remaining_deck_vals, n_sims):
    """Hit 시 각 시뮬레이션의 버스트 여부 반환"""
    if is_soft:
        return np.zeros(n_sims, dtype=np.int8)

    # 덱에서 무작위 추출 (복원 추출 근사)
    if len(remaining_deck_vals) == 0:
        return np.zeros(n_sims, dtype=np.int8)
    
    hit_cards = np.random.choice(remaining_deck_vals, size=n_sims, replace=True)
    # A는 1로 처리 가능
    hit_cards[hit_cards == 11] = 1
    
    bust_results = ((player_total + hit_cards) > 21).astype(np.int8)
    return bust_results

def simulate_dealer_turn_vectorized(dealer_up_card_id, player_total, remaining_deck_vals, n_sims):
    """딜러 턴 고속 시뮬레이션 - 각 시뮬레이션의 딜러 승리 여부 반환"""
    # 딜러 업 카드 값
    dealer_up_val = RANK_TO_VAL_INT[dealer_up_card_id[1:]]
    
    max_draws = 15 # 넉넉하게
    if len(remaining_deck_vals) == 0:
        return np.zeros(n_sims, dtype=np.int8)
    
    # 덱에서 무작위 추출 (복원 추출 근사)
    random_draws = np.random.choice(remaining_deck_vals, size=(n_sims, max_draws), replace=True)
    
    up_cards = np.full(n_sims, dealer_up_val, dtype=np.int8)
    hidden_cards = random_draws[:, 0]
    
    current_sums = up_cards + hidden_cards
    ace_counts = (up_cards == 11).astype(np.int8) + (hidden_cards == 11).astype(np.int8)
    
    mask_bust = (current_sums > 21) & (ace_counts > 0)
    current_sums[mask_bust] -= 10
    ace_counts[mask_bust] -= 1
    
    draw_idx = 1
    active_mask = current_sums < 17
    
    while np.any(active_mask) and draw_idx < max_draws:
        new_cards = random_draws[:, draw_idx]
        
        current_sums[active_mask] += new_cards[active_mask]
        ace_counts[active_mask] += (new_cards[active_mask] == 11).astype(np.int8)
        
        bust_with_ace = (current_sums > 21) & (ace_counts > 0)
        current_sums[bust_with_ace] -= 10
        ace_counts[bust_with_ace] -= 1
        
        # Double check for multiple aces case
        bust_with_ace_2 = (current_sums > 21) & (ace_counts > 0)
        current_sums[bust_with_ace_2] -= 10
        ace_counts[bust_with_ace_2] -= 1
        
        active_mask = current_sums < 17
        draw_idx += 1
    
    dealer_wins = ((current_sums <= 21) & (current_sums > player_total)).astype(np.int8)
    
    return dealer_wins

def generate_row_data(player_cards_ids, dealer_card_id, bust_results, dealer_win_results):
    """시뮬레이션 결과를 집계하여 하나의 row 생성 (확률 포함)"""
    # 1. 현재 오픈된 카드 벡터 (Current: 52 cols)
    current_vec = np.zeros(52, dtype=int)
    used_cards = player_cards_ids + [dealer_card_id]
    
    for cid in used_cards:
        idx = CARD_TO_IDX[cid]
        current_vec[idx] += 1 # One-hot 대신 Count (같은 카드가 2장일 수 있으므로)
        
    # 2. 남은 카드 벡터 (Remaining: 52 cols)
    # 전체 6덱 상태에서 used_cards를 뺀 것
    # 6덱 기준 각 카드는 6장씩 있음
    remaining_vec = np.full(52, DECK_COUNT, dtype=int) 
    remaining_vec -= current_vec
    
    # 3. 확률 계산
    hit_bust_prob = np.mean(bust_results)
    stay_dealer_win_prob = np.mean(dealer_win_results)
    
    # 4. 하나의 row 생성
    row = {}
    
    # Remaining Columns
    for j, cid in enumerate(ALL_CARD_IDS):
        row[f'rem_{cid}'] = remaining_vec[j]
        
    # Current Columns
    for j, cid in enumerate(ALL_CARD_IDS):
        row[f'curr_{cid}'] = current_vec[j]
        
    # Output Columns (확률)
    row['hit_bust_prob'] = float(hit_bust_prob)
    row['stay_dealer_win_prob'] = float(stay_dealer_win_prob)
    
    return row

def save_batch_to_csv(batch_results, file_path, cols_order, is_first_batch=False):
    """배치 결과를 CSV 파일에 저장 (첫 배치는 헤더 포함, 이후는 append)"""
    if not batch_results:
        return
    
    df_batch = pd.DataFrame(batch_results)
    df_batch = df_batch[cols_order]  # 순서 정렬
    
    mode = 'w' if is_first_batch else 'a'
    header = is_first_batch
    df_batch.to_csv(file_path, mode=mode, index=False, header=header)

def run_ml_data_generation():
    start_time = time.time()
    batch_results = []  # 배치 단위로 저장할 결과
    
    print(f"Starting ML Data Generation (Probabilities)...")
    print(f"- Output Columns: 104 Inputs (rem_* 52 + curr_* 52) + 2 Outputs (hit_bust_prob, stay_dealer_win_prob)")
    print(f"- Simulation: {SIMULATION_COUNT} iter/case")
    print(f"- Saving aggregated probabilities (one row per scenario)")
    print(f"- Batch size: {BATCH_SIZE} scenarios per save")
    
    # 컬럼 순서 정의
    cols_order = []
    for cid in ALL_CARD_IDS: cols_order.append(f'rem_{cid}')
    for cid in ALL_CARD_IDS: cols_order.append(f'curr_{cid}')
    cols_order.extend(['hit_bust_prob', 'stay_dealer_win_prob'])
    
    # 확률 계산용 기본 값 덱
    base_full_deck_vals = create_full_deck_values_array()
    
    dealer_ranks_loop = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
    
    total_scenarios = 0
    total_rows_saved = 0
    is_first_batch = True
    
    # 기존 파일이 있으면 삭제 (새로 시작)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    # 1. Hard Hands (8 ~ 20)
    for total in range(8, 21):
        for d_rank in dealer_ranks_loop:
            total_scenarios += 1
            # -------------------------------------------------
            # 구체적인 카드 조합 생성 (Random Suit)
            # -------------------------------------------------
            # 딜러 카드 (무늬 랜덤)
            d_suit = random.choice(SUITS)
            d_card_id = f"{d_suit}{d_rank}"
            
            # 플레이어 카드 (Total을 만족하는 구성 생성)
            # 단순화를 위해 2장으로 구성 (Hard Hand)
            p_cards_ids = []
            if total <= 10:
                # [2, Total-2]
                r1, r2 = '2', str(total - 2)
            elif total == 11:
                # 11은 [2, 9] 또는 [3, 8] 등으로 구성 (1은 유효한 랭크가 아님)
                r1, r2 = '2', '9'
            elif total == 20:
                r1, r2 = '10', '10' # 10, K 등 다양하지만 단순화
            else:
                # [10, Total-10]
                r1, r2 = '10', str(total - 10)
            
            # 무늬 랜덤 할당 (중복 방지 로직은 간단히 생략 - 6덱이라 충돌 확률 낮고 괜찮음)
            p_cards_ids.append(f"{random.choice(SUITS)}{r1}")
            p_cards_ids.append(f"{random.choice(SUITS)}{r2}")
            
            # -------------------------------------------------
            # 시뮬레이션 실행
            # -------------------------------------------------
            used_ids = p_cards_ids + [d_card_id]
            rem_deck_vals = get_remaining_deck_values(base_full_deck_vals, used_ids)
            
            bust_results = calc_bust_results(total, False, rem_deck_vals, SIMULATION_COUNT)
            dealer_win_results = simulate_dealer_turn_vectorized(d_card_id, total, rem_deck_vals, SIMULATION_COUNT)
            
            # -------------------------------------------------
            # 시뮬레이션 결과를 집계하여 하나의 row 생성
            # -------------------------------------------------
            row = generate_row_data(p_cards_ids, d_card_id, bust_results, dealer_win_results)
            batch_results.append(row)
            
            # 배치 크기에 도달하면 파일에 저장
            if total_scenarios % BATCH_SIZE == 0:
                save_batch_to_csv(batch_results, OUTPUT_FILE, cols_order, is_first_batch)
                total_rows_saved += len(batch_results)
                print(f"Processed {total_scenarios} scenarios, saved {total_rows_saved} rows so far...")
                batch_results = []  # 배치 초기화
                is_first_batch = False

    # 2. Soft Hands (13 ~ 20 -> A+2 ~ A+9)
    for other in range(2, 10):
        total = 11 + other
        for d_rank in dealer_ranks_loop:
            total_scenarios += 1
            # 딜러 카드
            d_suit = random.choice(SUITS)
            d_card_id = f"{d_suit}{d_rank}"
            
            # 플레이어 카드 (Ace + Other)
            p_cards_ids = []
            p_cards_ids.append(f"{random.choice(SUITS)}A")
            p_cards_ids.append(f"{random.choice(SUITS)}{other}")
            
            # 시뮬레이션 실행
            used_ids = p_cards_ids + [d_card_id]
            rem_deck_vals = get_remaining_deck_values(base_full_deck_vals, used_ids)
            
            bust_results = calc_bust_results(total, True, rem_deck_vals, SIMULATION_COUNT) # is_soft=True
            dealer_win_results = simulate_dealer_turn_vectorized(d_card_id, total, rem_deck_vals, SIMULATION_COUNT)
            
            row = generate_row_data(p_cards_ids, d_card_id, bust_results, dealer_win_results)
            batch_results.append(row)
            
            # 배치 크기에 도달하면 파일에 저장
            if total_scenarios % BATCH_SIZE == 0:
                save_batch_to_csv(batch_results, OUTPUT_FILE, cols_order, is_first_batch)
                total_rows_saved += len(batch_results)
                print(f"Processed {total_scenarios} scenarios, saved {total_rows_saved} rows so far...")
                batch_results = []  # 배치 초기화
                is_first_batch = False
    
    # 남은 배치 저장
    if batch_results:
        save_batch_to_csv(batch_results, OUTPUT_FILE, cols_order, is_first_batch)
        total_rows_saved += len(batch_results)
    
    end_time = time.time()
    print(f"\nGenerated {total_rows_saved} rows with {len(cols_order)} columns.")
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Elapsed: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    run_ml_data_generation()

