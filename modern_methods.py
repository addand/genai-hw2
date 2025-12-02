import google.generativeai as genai
import re
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pandas as pd

#使用 Google Gemini API 設定
try:
    GOOGLE_API_KEY = "XXX"
    genai.configure(api_key=GOOGLE_API_KEY)
    print("✅ API Key 設定成功")
except Exception as e:
    print(f"API Key 設定失敗: {e}")
    

#B-1:語意相似度計算
# B-1: 語意相似度計算
def ai_similarity(text1, text2):
    prompt = f"""
    請擔任一位語意分析專家，評估以下兩段文字的語意相似度。

    考慮因素：
    1. 主題相關性
    2. 語意重疊程度
    3. 觀點一致性

    文字1: {text1}
    文字2: {text2}

    請輸出一個 0 到 100 的整數分數。

    Constraints:
    - 嚴格只輸出數字 (例如: 85)。
    - 不要輸出任何解釋。
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        generation_config = genai.types.GenerationConfig(temperature=0.0)

        response = model.generate_content(prompt, generation_config=generation_config)
        result_text = response.text.strip()

        match = re.search(r'\d+', result_text)
        if match:
            score = int(match.group())
            return max(0, min(100, score))
        else:
            print(f"Parsing Error: Model output '{result_text}'")
            return -1

    except Exception as e:
        print(f"API Error: {e}")
        return -1


import time
import os
import numpy as np
import matplotlib.pyplot as plt

def run_test_and_dump_b1():
    print("=== B1: AI 語意相似度 (含成本與時間計算) ===")

    # 1. 準備資料
    texts = [
        "人工智慧正在改變世界， 機器學習是其核心技術",
        "深度學習推動了人工智慧的發展， 特別是在圖像識別領域",
        "今天天氣很好， 適合出去運動",
        "機器學習和深度學習都是人工智慧的重要分支",
        "運動有益健康， 每天都應該保持運動習慣"
    ]

    n = len(texts)
    labels = [f"T{i+1}" for i in range(n)]

    # === 成本設定 (每 100 萬 Token 的價格) ===
    # 依據您提供的費率：
    PRICE_INPUT_PER_1M = 0.10   # 輸入: $0.10 / 1M tokens
    PRICE_OUTPUT_PER_1M = 0.40  # 輸出: $0.40 / 1M tokens

    # 2. 變數初始化
    sim_matrix = np.zeros((n, n), dtype=float)
    
    total_input_tokens = 0   # 累計輸入 Token
    total_output_tokens = 0  # 累計輸出 Token
    api_call_count = 0       # 實際呼叫次數

    print(f"開始計算相似度矩陣 (共 {n}x{n} 格，扣除重複與對角線)...")
    
    # 【開始計時】
    start_time = time.time()

    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 100.0
            elif j < i:
                sim_matrix[i, j] = sim_matrix[j, i]
            else:
                # === 模擬 API 呼叫 ===
                
                # 1. 計算輸入 Token (Input)
                # 邏輯：Prompt 包含兩段文字，所以是 len(A) + len(B)
                # 註：這裡假設 1 個中文字元約等於 1 個 Token (簡易估算)
                current_input_tokens = len(texts[i]) + len(texts[j])
                total_input_tokens += current_input_tokens
                
                # 2. 計算輸出 Token (Output)
                # 邏輯：API 回傳一個分數 (例如 "85" 或 JSON)，通常很短，估算約 5 tokens
                current_output_tokens = 5 
                total_output_tokens += current_output_tokens
                
                api_call_count += 1
                
                # 3. 執行函式
                score = ai_similarity(texts[i], texts[j])
                if score < 0: score = 0
                sim_matrix[i, j] = score

    # 【結束計時】
    end_time = time.time()
    total_duration = end_time - start_time

    # === 成本計算公式 ===
    # 成本 = (Token數 / 1,000,000) * 百萬Token單價
    cost_input = (total_input_tokens / 1_000_000) * PRICE_INPUT_PER_1M
    cost_output = (total_output_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M
    total_cost_usd = cost_input + cost_output
    total_cost_twd = total_cost_usd * 32.5  # 假設匯率 1:32.5

    print("-" * 40)
    print(f"計算完成！統計結果如下：")
    print(f"★ API 處理總耗時 : {total_duration:.4f} 秒")
    print(f"★ 實際呼叫次數   : {api_call_count} 次")
    print("-" * 40)
    print(f"【Token 統計】")
    print(f"  - 輸入 Tokens  : {total_input_tokens}")
    print(f"  - 輸出 Tokens  : {total_output_tokens}")
    print(f"  - 總計 Tokens  : {total_input_tokens + total_output_tokens}")
    print("-" * 40)
    print(f"【預估成本】(Input ${PRICE_INPUT_PER_1M}/1M, Output ${PRICE_OUTPUT_PER_1M}/1M)")
    print(f"  - 輸入成本     : ${cost_input:.8f} USD")
    print(f"  - 輸出成本     : ${cost_output:.8f} USD")
    print(f"  - 總計美金     : ${total_cost_usd:.8f} USD")
    print("-" * 40)

    # 3. 繪圖 (保持原樣)
    os.makedirs("results", exist_ok=True)
    # 繪製熱點圖
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, interpolation="nearest", cmap="viridis")
    plt.title("AI Semantic Similarity Matrix")
    plt.colorbar(label="Semantic similarity (0-100)")
    ticks = range(n)
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.savefig("results/ai_similarity_matrix.png", dpi=300)
    plt.close()

#B-2: AI 文本分類
def batch_ai_classify(batch_data):
    """
    使用 Gemini 進行批次文本分類
    Args:
        batch_data (list): 包含 [{'id': index, 'text': text}, ...] 的列表
    Returns:
        dict: Key為id, Value為 {'sentiment': ..., 'topic': ...} 的字典
    """
    
    # 1. 模型設定
    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash-lite', # 使用最新的 flash 模型
            generation_config={
                "response_mime_type": "application/json", # 強制 JSON
                "temperature": 0.1
            }
        )
    except Exception as e:
        print(f"模型初始化失敗: {e}")
        return {}

    # 2. 建構 Prompt (要求一次回傳一個 List)
    input_json_str = json.dumps(batch_data, ensure_ascii=False)
    
    prompt = f"""
    你是一個批次文本分類系統。請分析輸入數據中的每一條文本。
    
    輸入數據 (JSON List):
    {input_json_str}

    請回傳一個 JSON List，其中包含每個項目的分析結果。
    每個結果物件必須包含：
    1. "id": 對應輸入數據的 id (必須完全一致)
    2. "sentiment": 情感判斷 (只能是: "正面", "負面", "中性")
    3. "topic": 主題類別，主題類別以以下關鍵字來挑選，只能從科技運動美食旅遊娛樂來擇一
    '科技': 關鍵字['AI', '人工智慧', '電腦', '軟體', '程式', '演算法', '深度學習', '技術', '家電'],
    '運動': 關鍵字['運動', '健身', '跑步', '游泳', '球類', '比賽', '慢跑', '重訓', '體能'],
    '美食': 關鍵字['吃', '食物', '餐廳', '美味', '料理', '烹飪', '好吃', '湯頭', '麵條'],
    '旅遊': 關鍵字['旅行', '景點', '飯店', '機票', '觀光', '度假'],
    '娛樂': 關鍵字['電影', '劇情', '演技', '音樂', '遊戲']
    請確保回傳的列表長度與輸入一致。
    """

    try:
        # 3. 呼叫 API
        response = model.generate_content(prompt)
        result_list = json.loads(response.text)
        
        # 4. 轉換為以 ID 為 Key 的字典，方便後續對應
        # 格式: { 0: {'sentiment': '正面', 'topic': '科技'}, ... }
        result_map = {}
        for item in result_list:
            if 'id' in item:
                result_map[item['id']] = {
                    'sentiment': item.get('sentiment', '未知'),
                    'topic': item.get('topic', '未知')
                }
        return result_map

    except Exception as e:
        print(f"Batch API Error: {e}")
        # 如果解析失敗，試著印出原始回傳以便除錯
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Raw Response: {response.text[:200]}...")
        return {} 

# ==========================================
# 主程式：讀取 CSV -> 批次處理 -> 寫回 CSV
# ==========================================
def run_batch_classification_b2():
    print("=== B2 AI 批次分類任務 ===")
    
    # --- 1. 設定路徑 (本機端專用寫法) ---
    # __file__ 代表目前這隻程式的位置，這樣寫可以確保在任何目錄執行都能找到相對應的檔案
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "results", "classification_result.csv")

    # --- 2. 讀取 CSV ---
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 {file_path}")
        print(f"請確認檔案是否位於: {file_path}")
        return

    print(f"正在讀取檔案: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig') # 優先讀取 utf-8-sig
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='cp950') # 備用 Big5 (Excel 預設)
        except Exception as e:
            print(f"讀取 CSV 失敗，請檢查編碼: {e}")
            return

    # --- 3. 準備欄位 (安全寫法) ---
    # 確保 DataFrame 至少有 5 欄 (0:Input, 1:Rule_Sent, 2:Rule_Topic, 3:AI_Sent, 4:AI_Topic)
    required_cols = 5
    current_cols = df.shape[1]
    
    if current_cols < required_cols:
        for i in range(current_cols, required_cols):
            df.insert(i, f"New_Col_{i}", "")
    
    # 重新命名欄位 (直接修改 columns 列表)
    new_columns = list(df.columns)
    if len(new_columns) > 3: new_columns[3] = "AI_Sentiment"
    if len(new_columns) > 4: new_columns[4] = "AI_Topic"
    df.columns = new_columns

    # --- 4. 批次處理邏輯 ---
    BATCH_SIZE = 10 # 每次處理 10 筆
    total_rows = len(df)
    
    # 用來暫存要更新的數據 (初始化為空字串)
    ai_sentiments = [""] * total_rows
    ai_topics = [""] * total_rows

    # 取得要分析的文字欄位 (假設是第 0 欄)
    text_col_idx = 0 
    # 【開始計時】
    start_time = time.time()
    print(f"開始批次處理，共 {total_rows} 筆資料...")

    for i in range(0, total_rows, BATCH_SIZE):
        # 取出一批次資料
        batch_df = df.iloc[i : i + BATCH_SIZE]
        
        # 準備給 API 的資料格式: [{'id': 0, 'text': '...'}, ...]
        batch_input = []
        for idx, row in batch_df.iterrows():
            batch_input.append({
                "id": int(idx),  # 強制轉為 Python int
                "text": str(row.iloc[text_col_idx])
            })
        
        print(f"  - 正在處理 Index {i} 到 {min(i+BATCH_SIZE, total_rows)-1} ...")
        
        # 呼叫 API
        results_map = batch_ai_classify(batch_input)
        
        # 將結果填回暫存 List
        for idx, _ in batch_df.iterrows():
            idx = int(idx) # 確保索引一致
            if idx in results_map:
                ai_sentiments[idx] = results_map[idx]['sentiment']
                ai_topics[idx] = results_map[idx]['topic']
            else:
                ai_sentiments[idx] = "Error"
                ai_topics[idx] = "Error"
        
        # 簡單的 Rate Limiting，避免觸發 API 限制
        time.sleep(1)

    # 【結束計時】
    end_time = time.time()
    total_duration = end_time - start_time

    # --- 5. 更新 DataFrame ---
    # 將結果寫入第 3 欄 (D) 和 第 4 欄 (E)
    df.iloc[:, 3] = ai_sentiments
    df.iloc[:, 4] = ai_topics

    # --- 6. 存檔 ---
    output_path = file_path # 覆蓋原檔
    try:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n處理完成！結果已存回：{output_path}!")
        print(f"★ 任務總耗時: {total_duration:.2f} 秒")
    except Exception as e:
        print(f"存檔失敗: {e}")
        # 萬一存檔失敗 (例如檔案被 Excel 開啟中)，嘗試存到新檔名
        backup_path = file_path.replace(".csv", "_backup.csv")
        df.to_csv(backup_path, index=False, encoding='utf-8-sig')
        print(f"已嘗試存至備份檔: {backup_path}")


#B-3: AI 自動摘要
def ai_summarize(text, max_length):
    """
    使用 Gemini 生成指定長度的文本摘要
    Args:
        text (str): 原始長篇文章
        max_length (int): 期望的摘要字數限制
    Returns:
        str: 生成的摘要內容
    """

    # 1. 模型設定
    model = genai.GenerativeModel(
        'gemini-2.5-flash-lite',
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": max_length * 2
        }
    )

    # 2. Prompt
    prompt = f"""
    任務：請擔任專業編輯，將以下文章進行重點摘要。

    要求：
    1. 保留最關鍵的資訊
    2. 確保語句通順流暢，適合閱讀。
    3. 嚴格控制長度：請將摘要控制在 {max_length} 個字以內。

    原始文章：
    "{text}"

    摘要結果：
    """

    try:
        # 3. 生成內容
        response = model.generate_content(prompt)

        # 4. 結果處理
        if response.text:
            return response.text.strip()
        else:
            return "Error: Empty response from model"

    except Exception as e:
        return f"System Error: {str(e)}"

def run_test_and_dump_b3():
    print("=== B3 ===")
    text_data = """
    人工智慧(AI)的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘，到通勤時的路線規劃，再到工作中的各種輔助工具，AI無處不在。
    在醫療領域，AI協助醫生進行疾病診斷，提高了診斷的準確率和效率。透過分析大量的醫療影像和病歷資料，AI能夠發現人眼容易忽略的細節，為患者提供更好的治療方案。
    教育方面，AI個人化學習系統能夠根據每個學生的學習進度和特點，提供客製化的教學內容。這種因材施教的方式，讓學習變得更加高效和有趣。
    然而，AI的快速發展也帶來了一些挑戰。首先是就業問題，許多傳統工作可能會被AI取代。其次是隱私和安全問題，AI系統需要大量數據來訓練，如何保護個人隱私成為重要議題。
    最後是倫理問題，AI的決策過程往往缺乏透明度，可能會產生偏見或歧視。面對這些挑戰，我們需要在推動AI發展的同時，建立相應的法律法規和倫理準則。
    只有這樣，才能確保AI技術真正為人類福祉服務，創造一個更美好的未來。
    """

    # 設定目標長度為 50 字
    target_length = 50

    print(f"原始文章長度: {len(text_data)} 字")
    print(f"目標摘要長度: {target_length} 字\n")
    summary = ai_summarize(text_data, target_length)

    print(f"摘要結果:\n{summary}")

    # 確保 results 資料夾存在
    os.makedirs("results", exist_ok=True)

    # 將 AI 總結結果寫入檔案
    output_path = os.path.join("results", "summarization_ai.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"AI 摘要已輸出到：{output_path}")



if __name__ == "__main__":
    run_test_and_dump_b1()
    run_batch_classification_b2()
    run_test_and_dump_b3()