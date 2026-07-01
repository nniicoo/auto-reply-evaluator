"""
客服自动回复质量评估系统
支持 LLM API 模式和 Mock 模式
"""

import json
import os
import sys

# 修复 Windows 控制台编码
sys.stdout.reconfigure(encoding='utf-8')
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime

# ============ 配置 ============

# 是否使用真实 LLM API（False = Mock 模式）
USE_LLM_API = False

# 如果使用真实 API，配置你的 API Key
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-xxx")

# ============ 评估指标定义 ============

METRICS = {
    "accuracy": {
        "name": "回复准确性",
        "weight": 0.4,
        "description": "回复内容是否准确回答了用户问题，是否存在事实错误或误导信息",
        "scoring_guide": {
            5: "完全准确，精准回答用户问题",
            4: "基本准确，有小的不精确之处",
            3: "部分准确，存在遗漏或模糊表述",
            2: "准确性不足，存在明显错误或误导",
            1: "严重不准确，可能造成用户误解"
        }
    },
    "completeness": {
        "name": "回复完整性",
        "weight": 0.35,
        "description": "回复是否覆盖了用户问题的所有要点，是否提供了充分的信息",
        "scoring_guide": {
            5: "完整覆盖所有要点，信息充分",
            4: "覆盖主要要点，有少量遗漏",
            3: "覆盖部分要点，关键信息有遗漏",
            2: "严重不完整，大部分要点未覆盖",
            1: "几乎未回答用户问题"
        }
    },
    "tone": {
        "name": "语气得体性",
        "weight": 0.25,
        "description": "回复语气是否专业友好，是否符合客服场景要求，是否有温度",
        "scoring_guide": {
            5: "专业友好，有温度，符合优质客服标准",
            4: "较为专业，语气尚可",
            3: "中规中矩，略显生硬",
            2: "语气冷淡或敷衍",
            1: "语气不当，可能引起用户不满"
        }
    }
}

# ============ 数据模型 ============

@dataclass
class EvalResult:
    id: int
    user_query: str
    auto_reply: str
    scores: Dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    suggestion: str = ""


# ============ Mock LLM 评估器 ============

class MockLLMEvaluator:
    """模拟 LLM 返回评分（用于无 API Key 时的演示）"""

    def __init__(self):
        # 预设的评分（基于对回复质量的人工判断）
        self.mock_scores = {
            1: {"accuracy": 4, "completeness": 3, "tone": 3, "issues": ["缺少主动查询承诺"], "suggestion": "增加主动帮用户查询物流的承诺"},
            2: {"accuracy": 4, "completeness": 2, "tone": 3, "issues": ["未说明保修范围和条件"], "suggestion": "补充保修范围、保修条件等信息"},
            3: {"accuracy": 5, "completeness": 5, "tone": 5, "issues": [], "suggestion": "保持当前水平"},
            4: {"accuracy": 4, "completeness": 4, "tone": 4, "issues": ["可补充仓库具体城市"], "suggestion": "增加具体仓库城市信息"},
            5: {"accuracy": 4, "completeness": 3, "tone": 3, "issues": ["未说明分期期数和免息活动"], "suggestion": "补充分期期数和免息活动信息"},
            6: {"accuracy": 5, "completeness": 4, "tone": 5, "issues": ["可补充运费承担说明"], "suggestion": "主动说明补发运费由商家承担"},
            7: {"accuracy": 4, "completeness": 4, "tone": 4, "issues": [], "suggestion": "保持当前水平"},
            8: {"accuracy": 5, "completeness": 4, "tone": 4, "issues": ["可主动询问订单状态"], "suggestion": "主动询问用户订单状态以便提供针对性建议"},
            9: {"accuracy": 2, "completeness": 1, "tone": 2, "issues": ["未提供任何有效信息", "反问式回复显得敷衍"], "suggestion": "应列出所有可选颜色及库存情况"},
            10: {"accuracy": 5, "completeness": 4, "tone": 4, "issues": ["可补充具体发货时间节点"], "suggestion": "补充每天的发货截止时间"},
            11: {"accuracy": 5, "completeness": 4, "tone": 4, "issues": ["可补充发票类型说明"], "suggestion": "补充支持的发票类型"},
            12: {"accuracy": 3, "completeness": 2, "tone": 2, "issues": ["回复过于简单", "未分析可能原因"], "suggestion": "分析可能的原因并提供排查建议"},
            13: {"accuracy": 4, "completeness": 3, "tone": 4, "issues": ["未说明需要提供哪些材料"], "suggestion": "列出需要用户提供的具体材料"},
            14: {"accuracy": 1, "completeness": 1, "tone": 1, "issues": ["回复只有'您好'，完全无效", "零信息量"], "suggestion": "需要重新生成，必须包含注册流程和会员权益"},
            15: {"accuracy": 5, "completeness": 5, "tone": 5, "issues": [], "suggestion": "保持当前水平"},
            16: {"accuracy": 4, "completeness": 4, "tone": 4, "issues": [], "suggestion": "保持当前水平"},
            17: {"accuracy": 4, "completeness": 2, "tone": 2, "issues": ["回复过于生硬", "未提供替代支付方式"], "suggestion": "补充支持的支付方式，语气更友好"},
            18: {"accuracy": 3, "completeness": 2, "tone": 3, "issues": ["空洞承诺可能无法兑现", "缺少具体订单状态"], "suggestion": "查询实际订单状态后再回复，避免空承诺"},
            19: {"accuracy": 4, "completeness": 3, "tone": 3, "issues": ["未提供正品验证方式"], "suggestion": "补充防伪查询、专柜验货等验证方式"},
            20: {"accuracy": 4, "completeness": 3, "tone": 3, "issues": ["缺少具体退款流程说明", "推卸感较强"], "suggestion": "提供详细退款流程，主动提出代为处理"}
        }

    def evaluate(self, item_id: int, user_query: str, auto_reply: str) -> dict:
        """模拟 LLM 评估"""
        if item_id in self.mock_scores:
            data = self.mock_scores[item_id]
            return {
                "accuracy": data["accuracy"],
                "completeness": data["completeness"],
                "tone": data["tone"],
                "issues": data["issues"],
                "suggestion": data["suggestion"]
            }
        # 默认中等评分
        return {
            "accuracy": 3,
            "completeness": 3,
            "tone": 3,
            "issues": ["无法评估"],
            "suggestion": "建议人工审核"
        }


# ============ 真实 LLM 评估器（需要 API Key）============

class LLMJudge:
    """使用真实 LLM API 进行评估"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        # 这里可以接入 OpenAI / Claude / 其他 LLM API
        # 示例使用 OpenAI 格式
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            print("请安装 openai: pip install openai")
            raise

    def evaluate(self, user_query: str, auto_reply: str) -> dict:
        prompt = f"""你是一个专业的客服质量评估专家。请对以下客服自动回复进行评分。

用户问题：{user_query}

自动回复：{auto_reply}

请从以下三个维度评分（1-5分）并给出分析：

1. **准确性**（accuracy）：回复是否准确回答了用户问题
2. **完整性**（completeness）：回复是否覆盖了所有要点
3. **语气得体性**（tone）：回复语气是否专业友好

请以JSON格式返回：
{{
    "accuracy": <分数>,
    "completeness": <分数>,
    "tone": <分数>,
    "issues": ["问题1", "问题2"],
    "suggestion": "改进建议"
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"API 调用失败: {e}")
            return {"accuracy": 0, "completeness": 0, "tone": 0, "issues": ["API错误"], "suggestion": "请重试"}


# ============ 评估流水线 ============

class EvalPipeline:
    def __init__(self, use_llm: bool = False, api_key: str = None):
        if use_llm and api_key:
            self.evaluator = LLMJudge(api_key)
            self.mode = "LLM API"
        else:
            self.evaluator = MockLLMEvaluator()
            self.mode = "Mock"

    def load_data(self):
        """加载数据文件"""
        with open("auto_replies.json", "r", encoding="utf-8") as f:
            self.auto_replies = json.load(f)
        with open("human_ref.json", "r", encoding="utf-8") as f:
            self.human_refs = json.load(f)

    def evaluate_single(self, item: dict) -> EvalResult:
        """评估单条回复"""
        result = EvalResult(
            id=item["id"],
            user_query=item["user_query"],
            auto_reply=item["auto_reply"]
        )

        if self.mode == "LLM API":
            scores = self.evaluator.evaluate(item["user_query"], item["auto_reply"])
        else:
            scores = self.evaluator.evaluate(item["id"], item["user_query"], item["auto_reply"])

        result.scores = {
            "accuracy": scores["accuracy"],
            "completeness": scores["completeness"],
            "tone": scores["tone"]
        }

        # 计算加权得分
        result.weighted_score = sum(
            scores[metric] * METRICS[metric]["weight"]
            for metric in METRICS
        )

        result.issues = scores.get("issues", [])
        result.suggestion = scores.get("suggestion", "")

        return result

    def run(self) -> List[EvalResult]:
        """运行完整评估流水线"""
        self.load_data()
        results = []

        print(f"\n{'='*60}")
        print(f"  客服自动回复质量评估系统")
        print(f"  评估模式: {self.mode}")
        print(f"  评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        for item in self.auto_replies:
            result = self.evaluate_single(item)
            results.append(result)
            print(f"✓ 已评估第 {result.id} 条回复 - 加权得分: {result.weighted_score:.2f}")

        return results

    def generate_report(self, results: List[EvalResult]) -> str:
        """生成评估报告"""
        report = []
        report.append("=" * 60)
        report.append("          客服自动回复质量评估报告")
        report.append("=" * 60)

        # 1. 整体统计
        total_weighted = sum(r.weighted_score for r in results) / len(results)
        avg_scores = {}
        for metric in METRICS:
            avg_scores[metric] = sum(r.scores[metric] for r in results) / len(results)

        report.append("\n【一、整体得分】")
        report.append(f"  综合加权得分: {total_weighted:.2f} / 5.00")
        report.append(f"  准确性均分:   {avg_scores['accuracy']:.2f} / 5.00")
        report.append(f"  完整性均分:   {avg_scores['completeness']:.2f} / 5.00")
        report.append(f"  语气得体均分: {avg_scores['tone']:.2f} / 5.00")

        # 2. 各指标分布
        report.append("\n【二、各指标得分分布】")
        for metric_key, metric_info in METRICS.items():
            report.append(f"\n  {metric_info['name']}（权重 {metric_info['weight']*100:.0f}%）:")
            scores = [r.scores[metric_key] for r in results]
            for score in range(5, 0, -1):
                count = scores.count(score)
                bar = "█" * count
                report.append(f"    {score}分: {bar} ({count}条)")

        # 3. 最差 3 条 case
        report.append("\n【三、最差 3 条 Case 分析】")
        worst_3 = sorted(results, key=lambda r: r.weighted_score)[:3]

        for i, r in enumerate(worst_3, 1):
            report.append(f"\n  --- 第 {i} 差 (ID: {r.id}) ---")
            report.append(f"  用户问题: {r.user_query}")
            report.append(f"  自动回复: {r.auto_reply}")
            report.append(f"  得分: 准确性={r.scores['accuracy']}, 完整性={r.scores['completeness']}, 语气={r.scores['tone']}")
            report.append(f"  加权得分: {r.weighted_score:.2f}")
            report.append(f"  问题: {', '.join(r.issues)}")
            report.append(f"  建议: {r.suggestion}")

            # 对比人工参考
            ref = next((h for h in self.human_refs if h["id"] == r.id), None)
            if ref:
                report.append(f"  参考回复: {ref['reference_reply'][:100]}...")

        # 4. 质量等级分布
        report.append("\n【四、质量等级分布】")
        excellent = sum(1 for r in results if r.weighted_score >= 4.5)
        good = sum(1 for r in results if 3.5 <= r.weighted_score < 4.5)
        average = sum(1 for r in results if 2.5 <= r.weighted_score < 3.5)
        poor = sum(1 for r in results if r.weighted_score < 2.5)

        report.append(f"  优秀 (≥4.5): {excellent} 条")
        report.append(f"  良好 (3.5-4.5): {good} 条")
        report.append(f"  一般 (2.5-3.5): {average} 条")
        report.append(f"  较差 (<2.5): {poor} 条")

        # 5. 结论与建议
        report.append("\n【五、结论与建议】")
        if total_weighted >= 4:
            report.append("  整体质量良好，建议保持并优化个别低分回复。")
        elif total_weighted >= 3:
            report.append("  整体质量中等，建议重点改进低分回复的完整性和语气。")
        else:
            report.append("  整体质量需要提升，建议重新优化自动回复模板。")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# ============ 主程序 ============

if __name__ == "__main__":
    pipeline = EvalPipeline(use_llm=USE_LLM_API)
    results = pipeline.run()

    report = pipeline.generate_report(results)
    print("\n" + report)

    # 保存报告到文件
    with open("eval_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已保存到 eval_report.txt")

    # 保存详细结果到 JSON
    detailed_results = []
    for r in results:
        detailed_results.append({
            "id": r.id,
            "user_query": r.user_query,
            "auto_reply": r.auto_reply,
            "scores": r.scores,
            "weighted_score": round(r.weighted_score, 2),
            "issues": r.issues,
            "suggestion": r.suggestion
        })

    with open("eval_results_detail.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    print("详细结果已保存到 eval_results_detail.json")
