from __future__ import annotations

from agile_bank.agents import copy
from agile_bank.agents.state import ConversationState
from agile_bank.agents.tools.interview import INTERVIEW_ORDER, _next_missing


def interview_prompt(state: ConversationState) -> str:
    interview = state.get("interview") or {}
    missing = _next_missing(interview)
    questions = "\n".join(
        f"- {field}: {copy.INTERVIEW_QUESTIONS[field]}"
        for field in INTERVIEW_ORDER
    )
    next_step = (
        f"Próxima pergunta: {copy.INTERVIEW_QUESTIONS[missing]}"
        if missing is not None
        else "Todas as respostas estão registradas; o sistema segue para a reanálise."
    )
    return (
        f"{copy.BASE_RULES}"
        "Nesta etapa você faz uma entrevista financeira curta para atualizar o "
        "score do cliente. Faça uma pergunta por mensagem, nesta ordem:\n"
        f"{questions}\n"
        "Depois de cada resposta do cliente, chame record_interview_answer uma "
        "vez, com o campo e o valor que ele acabou de dizer. O sistema aceita "
        "o próximo campo da ordem, ou a correção de um campo já gravado. "
        "Campo futuro é recusado. Valor que não cabe no campo também.\n"
        "- monthly_income e monthly_expenses: reais, só dígitos "
        "(R$ 10 mil vira 10000).\n"
        "- employment_type: formal (CLT, carteira assinada, servidor), autônomo "
        "(MEI, freelancer, por conta própria) ou desempregado.\n"
        "- dependents: número inteiro (nenhum vira 0).\n"
        "- has_debt: sim ou não.\n"
        "Se a resposta for ambígua, pergunte de novo citando as opções. Não "
        "comente as respostas e não dê conselho financeiro.\n"
        "Quando a última resposta válida for registrada, o sistema segue sozinho "
        "para a reanálise. Se ainda precisar, chame submit_interview sem escrever "
        "texto.\n"
        "Se o cliente perguntar para que serve, explique em uma frase que as "
        "respostas atualizam o score para reanalisar o pedido de aumento.\n"
        "Se o cliente desistir da entrevista e quiser falar de outro assunto, "
        "chame handoff_to_credit sem escrever texto.\n"
        f"{next_step}"
    )
