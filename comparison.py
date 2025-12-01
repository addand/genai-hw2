import pandas as pd
import os
import json
import time
import google.generativeai as genai

def calculate_extended_accuracy_report(file_path):
    print(f"正在評估檔案: {file_path}")
    
    try:
        # 讀取 CSV
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp950')
        except:
            df = pd.read_csv(file_path, encoding='utf-8')

        # 檢查欄位數量 (需至少 7 欄)
        if df.shape[1] < 7:
            print("錯誤：CSV 欄位不足，無法找到欄位 G。")
            return

        # 取得欄位資料 (轉字串並去空白，確保比對正確)
        # B: Rule-Based 情感, C: Rule-Based 主題
        # D: AI 情感, E: AI 主題
        # F: 真實情感, G: 真實主題
        col_b = df.iloc[:, 1].astype(str).str.strip()
        col_c = df.iloc[:, 2].astype(str).str.strip()
        col_d = df.iloc[:, 3].astype(str).str.strip()
        col_e = df.iloc[:, 4].astype(str).str.strip()
        col_f = df.iloc[:, 5].astype(str).str.strip()
        col_g = df.iloc[:, 6].astype(str).str.strip()
        
        total = len(df)
        if total == 0:
            print("錯誤：資料筆數為 0")
            return

        # 定義計算函式
        def get_accuracy(pred_col, true_col):
            matches = (pred_col == true_col).sum()
            return matches / total, matches

        # 1. 計算情感分析準確率
        acc_bf, match_bf = get_accuracy(col_b, col_f) # Rule-Based
        acc_df, match_df = get_accuracy(col_d, col_f) # AI Model

        # 2. 計算主題分類準確率
        acc_cg, match_cg = get_accuracy(col_c, col_g) # Rule-Based
        acc_eg, match_eg = get_accuracy(col_e, col_g) # AI Model

        # --- 輸出報告 ---
        print("-" * 45)
        print(f"【完整準確率評估報告】 總筆數: {total}")
        print("-" * 45)
        
        print(f"【情感分析 (Sentiment Analysis)】")
        print(f"  Ground Truth: 欄位 F (人類答案)")
        print(f"  -------------------------------------------")
        print(f"  1. 規則式 (欄位 B):  {acc_bf:.2%}  ({match_bf}/{total})")
        print(f"  2. AI模型 (欄位 D):  {acc_df:.2%}  ({match_df}/{total})")
        print("-" * 45)
        
        print(f"【主題分類 (Topic Classification)】")
        print(f"  Ground Truth: 欄位 G (人類分類)")
        print(f"  -------------------------------------------")
        print(f"  1. 規則式 (欄位 C):  {acc_cg:.2%}  ({match_cg}/{total})")
        print(f"  2. AI模型 (欄位 E):  {acc_eg:.2%}  ({match_eg}/{total})")
        print("-" * 45)

    except Exception as e:
        print(f"發生錯誤: {e}")


def generate_eval_prompt(original_text, ai_summary, tfidf_summary):
    return f"""
# Role
你是一位專業的 NLP 評估專家。請評估以下兩份摘要（TF-IDF 生成 vs AI 生成）的品質。

# Input Data
## [原文]
{original_text}

## [摘要 A: AI生成]
{ai_summary}

## [摘要 B: TF-IDF生成]
{tfidf_summary}

# Evaluation Metrics
1. **資訊保留度 (Information Retention)** (0-100%):
   - 摘要是否包含原文的關鍵事實與數據？
   - 忽略文法，只看「重點」是否還在。
   
2. **語句通順度 (Fluency)** (0-100%):
   - 語句是否通順、連貫？
   - 是否有斷詞破碎、邏輯不通的問題？

# Output Format (JSON Only)
請嚴格遵守以下 JSON 格式輸出：
{{
  "original_key_points_count": <整數: 原文關鍵點數量>,
  "ai_summary": {{
    "retention_score": <整數 0-100>,
    "fluency_score": <整數 0-100>,
    "missing_points": "<字串: 遺漏的關鍵點簡述>",
    "comment": "<字串: 簡短評語>"
  }},
  "tfidf_summary": {{
    "retention_score": <整數 0-100>,
    "fluency_score": <整數 0-100>,
    "missing_points": "<字串: 遺漏的關鍵點簡述>",
    "comment": "<字串: 簡短評語>"
  }},
  "winner": "<字串: 'AI' 或 'TF-IDF' 或 'Tie'>"
}}
"""

# ==========================================
# 2. Gemini API 呼叫實作
# ==========================================
def call_llm_api(prompt):
    """
    使用 Google Gemini API 進行評估。
    """
    # -----------------------------------------------------------
    # 【設定 API Key】請填入您的 Google AI Studio API Key
    # -----------------------------------------------------------
    API_KEY = "AIzaSyCfYZ65sWYK9ISihU7_x1me0V2MdCbVhNs"
    # 設定 Gemini
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel(
        'gemini-2.5-flash-lite',
        generation_config={"response_mime_type": "application/json"}
    )

    print("  [System] 正在呼叫 Gemini 2.5 Flash lite 進行評估...")
    
    try:
        # 發送請求
        response = model.generate_content(prompt)
        
        # 回傳生成的文字 (因為設定了 JSON mode，這裡拿到的就是純 JSON 字串)
        return response.text
        
    except Exception as e:
        print(f"  [Error] Gemini API 呼叫失敗: {e}")
        # 如果失敗，回傳一個空的 JSON 字串避免程式崩潰
        return "{}"

# ==========================================
# 3. 核心評估函式
# ==========================================
def run_batch_evaluation(original_text_data):
    print("=== 開始執行摘要評估任務 (Powered by Gemini) ===")

    # 設定檔案路徑
    base_dir = "results"
    path_ai = os.path.join(base_dir, "summarization_ai.txt")
    path_tfidf = os.path.join(base_dir, "summarization_tfidf.txt")

    # 1. 讀取檔案
    summaries = {}

    for name, path in [("AI", path_ai), ("TF-IDF", path_tfidf)]:
        if not os.path.exists(path) and os.path.exists(path + ".txt"):
            path += ".txt"
            
        if not os.path.exists(path):
            print(f"  [Error] 找不到檔案: {path}")
            print("  [Info] 因缺少必要檔案，無法進行評估，程式中止。")
            return

        with open(path, "r", encoding="utf-8") as f:
            summaries[name] = f.read().strip()
        print(f"  [OK] 成功讀取 {name} 摘要 (長度: {len(summaries[name])} 字)")

    # 2. 準備 Prompt
    print("  [Process] 正在生成評估 Prompt...")

    prompt = generate_eval_prompt(
        original_text=original_text_data,
        ai_summary=summaries["AI"],
        tfidf_summary=summaries["TF-IDF"]
    )

    # 3. 呼叫 Gemini LLM 進行評分
    try:
        response_str = call_llm_api(prompt)
        
        # 解析 JSON
        # 雖然 Gemini JSON mode 很穩，但還是做一下基本的清洗以防萬一
        cleaned_response = response_str.replace("```json", "").replace("```", "").strip()
        eval_result = json.loads(cleaned_response)
        
    except json.JSONDecodeError:
        print("  [Error] LLM 回傳的格式不是有效的 JSON")
        print("  Raw Response:", response_str)
        return
    except Exception as e:
        print(f"  [Error] 處理過程發生錯誤: {e}")
        return

    # 4. 輸出
    print("\n=== Gemini 評估結果 ===")
    print(json.dumps(eval_result, indent=2, ensure_ascii=False))

# ==========================================
# 執行區塊
# ==========================================
if __name__ == "__main__":
    calculate_extended_accuracy_report("results/classification_result.csv")
    original_text_source = """
    人工智慧(AI)的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘，到通勤時的路線規劃，再到工作中的各種輔助工具，AI無處不在。
    在醫療領域，AI協助醫生進行疾病診斷，提高了診斷的準確率和效率。透過分析大量的醫療影像和病歷資料，AI能夠發現人眼容易忽略的細節，為患者提供更好的治療方案。
    教育方面，AI個人化學習系統能夠根據每個學生的學習進度和特點，提供客製化的教學內容。這種因材施教的方式，讓學習變得更加高效和有趣。
    然而，AI的快速發展也帶來了一些挑戰。首先是就業問題，許多傳統工作可能會被AI取代。其次是隱私和安全問題，AI系統需要大量數據來訓練，如何保護個人隱私成為重要議題。
    最後是倫理問題，AI的決策過程往往缺乏透明度，可能會產生偏見或歧視。面對這些挑戰，我們需要在推動AI發展的同時，建立相應的法律法規和倫理準則。
    只有這樣，才能確保AI技術真正為人類福祉服務，創造一個更美好的未來。
    """

    # 確保資料夾存在
    os.makedirs("results", exist_ok=True)
    run_batch_evaluation(original_text_source)