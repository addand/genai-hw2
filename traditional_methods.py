import os
import re
import math
import jieba
import requests
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 忽略警告
warnings.filterwarnings("ignore")

# ==========================================
# 0. 全域環境初始化 (字型、詞庫、停用詞)
# ==========================================
def configure_matplotlib_chinese():
    """設定 Matplotlib 支援中文字型"""
    # 嘗試多種常見中文字型 (Windows: 微軟正黑體, Mac: Arial Unicode MS, Linux/Colab: 文泉驛正黑)
    font_list = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Zen Hei']
    plt.rcParams['font.sans-serif'] = font_list
    plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

def initialize_traditional_chinese():
    """
    下載並設定 jieba 為繁體中文模式，同時準備停用詞表 (具備自動備援機制)。
    """
    print("=== 初始化繁體中文環境 ===")

    # 1. 設定 jieba 繁體大詞庫
    dict_file = "dict.txt.big"
    if not os.path.exists(dict_file):
        print(f"正在下載繁體中文詞庫 ({dict_file})...")
        url = "https://raw.githubusercontent.com/fxsjy/jieba/master/extra_dict/dict.txt.big"
        try:
            r = requests.get(url, timeout=15)
            with open(dict_file, "wb") as f:
                f.write(r.content)
            print("詞庫下載完成。")
        except Exception as e:
            print(f"詞庫下載失敗，將使用預設詞庫 (效果可能較差)。錯誤: {e}")
    
    if os.path.exists(dict_file):
        jieba.set_dictionary(dict_file)
        print(f"已載入繁體詞庫: {dict_file}")

    # 2. 準備停用詞表 (優先下載 stopwords-iso，失敗則用內建)
    stop_file = "stopwords_zh.txt"
    stop_words = set()
    download_success = False

    if not os.path.exists(stop_file):
        print("正在下載通用中文停用詞表...")
        url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-zh/master/stopwords-zh.txt"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(stop_file, "w", encoding="utf-8") as f:
                    f.write(r.text)
                download_success = True
                print("停用詞表下載完成。")
        except Exception as e:
            print(f"外部停用詞下載失敗，切換至內建備援模式。")
    else:
        download_success = True

    # 3. 載入停用詞
    if download_success and os.path.exists(stop_file):
        with open(stop_file, "r", encoding="utf-8") as f:
            for line in f:
                stop_words.add(line.strip())
    else:
        # 內建備援清單 (如果網路不通，至少能過濾這些)
        basic_stops = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一個', '上', '也', '很', '到', '說', '要', '去', '你', '這', '那', '與', '為', '之', '而', '及', '著', '或是', '因為', '所以'}
        stop_words.update(basic_stops)

    # 4. 強制補強：台灣慣用語與標點符號 (無論下載是否成功都加入)
    taiwan_enhancements = {
        '我們', '你們', '他們', '大家', '自己', '這裡', '那裡', '這邊', '那邊',
        '的話', '接著', '然後', '於是', '關於', '至於', '以及', '並且',
        '其實', '基本上', '原則上', '總之', '看來', '一般', '通常', '來說',
        '部分', '方面', '時候', '結果', '目前', '現在', '已經', '未來',
        '表示', '認為', '覺得', '希望', '好像', '比如', '例如'
    }
    symbols = {'\n', '\r', ' ', '\t', '\u3000', '，', '。', '！', '？', '：', '；', '「', '」', '（', '）', '、'}
    
    stop_words.update(taiwan_enhancements)
    stop_words.update(symbols)
    
    print(f"停用詞載入完成，共 {len(stop_words)} 個詞彙。")
    return stop_words

# 執行初始化
configure_matplotlib_chinese()
STOP_WORDS = initialize_traditional_chinese()

# 確保結果目錄存在
os.makedirs("results", exist_ok=True)


# ==========================================
# A-1-2: TF-IDF 與 餘弦相似度 (視覺化)
# ==========================================
def run_test_and_dump_a1_2():
    print("\n=== A1-2: TF-IDF & Cosine Similarity ===")  
    
    # 記錄總開始時間 (使用 perf_counter 精度較高)
    t_total_start = time.perf_counter()

    documents = [
        "人工智慧正在改變世界，機器學習是其核心技術",
        "深度學習推動了人工智慧的發展，特別是在圖像識別領域",
        "今天天氣很好，適合出去運動",
        "機器學習和深度學習都是人工智慧的重要分支",
        "運動有益健康，每天都應該保持運動習慣"
    ]

    # --- 階段 1: 斷詞前處理 ---
    print("Step 1: 執行斷詞與過濾停用詞...")
    t_preprocess_start = time.perf_counter()
    
    documents_cut = []
    for doc in documents:
        # 使用全域繁體設定與停用詞
        words = [w for w in jieba.cut(doc) if w not in STOP_WORDS and w.strip()]
        documents_cut.append(" ".join(words))
    
    t_preprocess_end = time.perf_counter()
    preprocess_duration = t_preprocess_end - t_preprocess_start
    print(f"   [Log] 斷詞處理耗時: {preprocess_duration:.4f} 秒")

    # --- 階段 2: 數學運算 (TF-IDF + Cosine Similarity) ---
    print("Step 2: 建立矩陣與計算相似度...")
    t_calc_start = time.perf_counter()

    # 計算 TF-IDF
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
    tfidf_matrix = vectorizer.fit_transform(documents_cut)

    # 計算相似度
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    t_calc_end = time.perf_counter()
    calc_duration = t_calc_end - t_calc_start
    print(f"   [Log] 矩陣運算耗時: {calc_duration:.4f} 秒")

    # --- 總結算 ---
    total_processing_time = preprocess_duration + calc_duration
    print(f"=== 核心邏輯總處理時間: {total_processing_time:.4f} 秒 ===")

    # --- 階段 3: 視覺化
    t_plot_start = time.perf_counter()
    
    # 繪製熱點圖
    plt.figure(figsize=(8, 6))
    plt.imshow(cosine_sim, interpolation="nearest", cmap="viridis")
    plt.title(f"TF-IDF Similarity Matrix (Process Time: {total_processing_time:.3f}s)") # 標題加上時間
    plt.colorbar(label="Cosine similarity")

    ticks = range(len(documents))
    plt.xticks(ticks, [f"Doc {i+1}" for i in ticks])
    plt.yticks(ticks, [f"Doc {i+1}" for i in ticks])

    # 標註數值
    for i in range(len(documents)):
        for j in range(len(documents)):
            plt.text(j, i, f"{cosine_sim[i, j]:.2f}", 
                     ha="center", va="center", 
                     color="white" if cosine_sim[i, j] < 0.7 else "black")

    plt.tight_layout()
    output_path = "results/tfidf_similarity_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"熱點圖已儲存至: {output_path}")


# ==========================================
# A-2: 情感與主題分類 (Rule-Based)
# ==========================================
class RuleBasedSentimentClassifier:
    def __init__(self):
        self.positive_words = ['好', '棒', '優秀', '喜歡', '推薦', '滿意', '開心', '值得', '精彩', '完美']
        self.negative_words = ['差', '糟', '失望', '討厭', '不推薦', '浪費', '無聊', '爛', '糟糕', '差勁']
        self.negation_words = ['不', '沒', '無', '非', '別']
        self.degree_words = ['很', '非常', '超', '太', '真', '極']

    def classify(self, text):
        if not isinstance(text, str): return "中性"
        score = 0
        
        # 簡單的情感分數計算邏輯
        for word_list, weight in [(self.positive_words, 1), (self.negative_words, -1)]:
            for word in word_list:
                start_search = 0
                while True:
                    idx = text.find(word, start_search)
                    if idx == -1: break
                    current_score = weight
                    # 向前看2個字檢查否定詞或程度詞
                    look_back = text[max(0, idx-2): idx]
                    if any(neg in look_back for neg in self.negation_words): current_score *= -1
                    if any(deg in look_back for deg in self.degree_words): current_score *= 2
                    score += current_score
                    start_search = idx + 1

        if score > 0: return "正面"
        elif score < 0: return "負面"
        else: return "中性"

class TopicClassifier:
    def __init__(self):
        self.topic_keywords = {
            '科技': ['AI', '人工智慧', '電腦', '軟體', '程式', '演算法', '深度學習', '技術', '相機'],
            '運動': ['運動', '健身', '跑步', '游泳', '球類', '比賽', '慢跑', '重訓', '體能'],
            '美食': ['吃', '食物', '餐廳', '美味', '料理', '烹飪', '好吃', '湯頭', '麵條'],
            '旅遊': ['旅行', '景點', '飯店', '機票', '觀光', '度假'],
            '娛樂': ['電影', '劇情', '演技', '音樂', '遊戲']
        }

    def classify(self, text):
        if not isinstance(text, str): return "其他"
        scores = {topic: 0 for topic in self.topic_keywords}
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    scores[topic] += text.count(keyword)
        
        best_topic = max(scores, key=scores.get)
        return best_topic if scores[best_topic] > 0 else "其他"

def run_unified_classification():
    print("\n=== A2: 情感與主題分類任務 ===")
    
    sentiment_clf = RuleBasedSentimentClassifier()
    topic_clf = TopicClassifier()

    file_path = "results/classification_result.csv"
    
    # 建立測試資料 (如果檔案不存在)
    if not os.path.exists(file_path):
        print("建立測試用 CSV...")
        data = {
            'Text': [
                '這家餐廳的料理非常美味，湯頭很棒！',
                '今天天氣太糟糕了，一直下雨好討厭。',
                '深度學習是目前人工智慧最熱門的技術。',
                '這部電影劇情很無聊，完全浪費時間。',
                '我喜歡每天去公園跑步健身。'
            ]
        }
        pd.DataFrame(data).to_csv(file_path, index=False, encoding='utf-8-sig')

    # 讀取 CSV
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')

    col_name = df.columns[0]
    
    # 執行分類
    df['情感預測'] = df[col_name].astype(str).apply(sentiment_clf.classify)
    df['主題預測'] = df[col_name].astype(str).apply(topic_clf.classify)

    # 存檔 (使用 utf-8-sig 以便 Excel 開啟)
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    print(f"分類結果已儲存至：{file_path}")
    print(df.head())


# ==========================================
# A-3: 統計式自動摘要
# ==========================================
class StatisticalSummarizer:
    def __init__(self):
        self.stop_words = STOP_WORDS

    def sentence_score(self, sentence, word_freq, index=0, total_sentences=1):
        words = [w for w in jieba.cut(sentence) if w.strip()]
        if not words: return 0

        score = 0
        # 1. 詞頻加總
        for w in words:
            if w not in self.stop_words:
                score += word_freq.get(w, 0)

        # 2. 位置權重
        if index == 0: score *= 1.5
        elif index == total_sentences - 1: score *= 1.2

        # 3. 長度權重
        if len(sentence) < 5: score *= 0.5
        elif len(sentence) > 50: score *= 0.8

        # 4. 數字加權
        if any(char.isdigit() for char in sentence): score *= 1.2

        return score

    def summarize(self, text, ratio=0.3):
        # 分句
        raw_sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        if not sentences: return ""

        # 計算總詞頻
        all_words = []
        for s in sentences:
            words = [w for w in jieba.cut(s) if w not in self.stop_words and w.strip()]
            all_words.extend(words)
        word_freq = Counter(all_words)

        # 計算句子分數
        ranked_sentences = []
        for i, s in enumerate(sentences):
            sc = self.sentence_score(s, word_freq, index=i, total_sentences=len(sentences))
            ranked_sentences.append((i, s, sc))

        # 排序選取
        num_sentences = max(1, int(len(sentences) * ratio))
        top_sentences = sorted(ranked_sentences, key=lambda x: x[2], reverse=True)[:num_sentences]
        # 還原順序
        top_sentences = sorted(top_sentences, key=lambda x: x[0])

        return "。".join([item[1] for item in top_sentences]) + "。"

def run_test_and_dump_a3():
    print("\n=== A3: 統計式自動摘要 ===")
    
    text_data = """
    人工智慧(AI)的發展正在深刻改變我們的生活方式。從早上起床時的智慧鬧鐘，到通勤時的路線規劃，再到工作中的各種輔助工具，AI無處不在。
    在醫療領域，AI協助醫生進行疾病診斷，提高了診斷的準確率和效率。透過分析大量的醫療影像和病歷資料，AI能夠發現人眼容易忽略的細節，為患者提供更好的治療方案。
    教育方面，AI個人化學習系統能夠根據每個學生的學習進度和特點，提供客製化的教學內容。這種因材施教的方式，讓學習變得更加高效和有趣。
    然而，AI的快速發展也帶來了一些挑戰。首先是就業問題，許多傳統工作可能會被AI取代。其次是隱私和安全問題，AI系統需要大量數據來訓練，如何保護個人隱私成為重要議題。
    最後是倫理問題，AI的決策過程往往缺乏透明度，可能會產生偏見或歧視。面對這些挑戰，我們需要在推動AI發展的同時，建立相應的法律法規和倫理準則。
    只有這樣，才能確保AI技術真正為人類福祉服務，創造一個更美好的未來。
    """

    summarizer = StatisticalSummarizer()
    result = summarizer.summarize(text_data, ratio=0.35) # 取約 35% 的內容

    print("【原文長度】:", len(text_data))
    print("【摘要結果】:\n", result)

    output_path = "results/summarization_tfidf.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"摘要已輸出到：{output_path}")

# ==========================================
# 主程式執行入口
# ==========================================
if __name__ == "__main__":
    run_test_and_dump_a1_2()
    run_unified_classification()
    run_test_and_dump_a3()