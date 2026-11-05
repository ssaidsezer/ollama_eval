from __future__ import annotations


DEFAULT_EVAL_SYSTEM_PROMPT = (
    "Sen taktik muharebe ve askeri eğitim alanında uzman bir otomatik değerlendirici modelsin.\n"
    "Aşağıdaki kurallara kesinlikle uy:\n"
    "- Yalnızca tek bir düz JSON nesnesi döndür; başka hiçbir metin ekleme.\n"
    "- Tüm string değerleri Türkçe yaz.\n"
    "- 'ai_verdict' için SADECE şu değerlerden birini kullan: correct | partial | incorrect\n"
    "  * correct: cevap beklenenle büyük ölçüde örtüşüyor, kritik bilgi eksik değil\n"
    "  * partial: cevap kısmen doğru fakat önemli detay eksik ya da yanıltıcı\n"
    "  * incorrect: cevap yanlış, tehlikeli veya tamamen alakasız\n"
    "- 'ai_score' 0-10 arası tam sayı olsun (10=mükemmel, 5=kısmen doğru, 0=yanlış/tehlikeli)\n"
    "- 'ai_hallucination_risk' için SADECE şu değerlerden birini kullan: low | medium | high\n"
    "- 'ai_reason' en fazla 2 kısa Türkçe cümle; neyin doğru/yanlış olduğunu açıkla\n\n"
    "Döndürülecek JSON şeması (başka alan ekleme):\n"
    "{\n"
    '  "ai_verdict": "correct" veya "partial" veya "incorrect",\n'
    '  "ai_score": 0-10 tam sayı,\n'
    '  "ai_hallucination_risk": "low" veya "medium" veya "high",\n'
    '  "ai_reason": string\n'
    "}\n"
)


def build_rag_prompt(question: str, context: str) -> str:
    """
    Turkish RAG prompt: prioritize context, then fallback to general knowledge.
    """
    if not context.strip():
        return (
            "Soruyu Türkçe ve kısa şekilde yanıtla.\n"
            "Yanıtında kesinlikle Markdown kullanma: tablo, başlık, bold, italik, madde imi veya herhangi bir formatlama işareti olmasın. Yanıt düz metin olarak seslendirilecek.\n"
            "Bu soruya ait bağlam verilmediği için genel bilgini kullanabilirsin.\n"
            "Emin olmadığın noktada varsayım uydurma; belirsiz olduğunu açıkça belirt.\n\n"
            f"Soru:\n{question}\n\n"
            "Cevap:\n"
        )

    return (
        "Soruyu Türkçe ve kısa şekilde yanıtla.\n"
        "Yanıtında kesinlikle Markdown kullanma: tablo, başlık, bold, italik, madde imi veya herhangi bir formatlama işareti olmasın. Yanıt düz metin olarak seslendirilecek.\n"
        "Önceliğin aşağıdaki bağlamdır; önce bağlamdaki bilgiyi kullan.\n"
        "Bağlam soruyu tam karşılamıyorsa, genel bilgini kullanarak cevaplayabilirsin.\n"
        "Bağlamla çelişen bir iddia üretme; emin olmadığın noktada bunu açıkça belirt.\n\n"
        f"Soru:\n{question}\n\n"
        f"Bağlam:\n{context}\n\n"
        "Cevap:\n"
    )


def build_no_rag_prompt(question: str) -> str:
    """
    Simple Turkish non-RAG prompt: answer from general knowledge.
    """
    return (
        "Sen Türkçe konuşan bir uzmansın.\n"
        "Yanıtında kesinlikle Markdown kullanma: tablo, başlık, bold, italik, madde imi veya herhangi bir formatlama işareti olmasın. Yanıt düz metin olarak seslendirilecek.\n"
        "Aşağıdaki soruyu kendi bilginle, net ve kısa biçimde cevapla.\n"
        "Cevaptan emin değilsen dürüst ol ve uydurma.\n\n"
        f"Soru: {question}\n\n"
        "Cevabın:\n"
    )
