from __future__ import annotations

from agile_bank.agents import copy
from agile_bank.agents.state import ConversationState


def credit_prompt(state: ConversationState) -> str:
    open_request = "sim" if state.get("open_limit_request") else "não"
    return (
        f"{copy.BASE_RULES}"
        "Nesta etapa você cuida do limite de crédito do cliente, que já está "
        "autenticado. Não peça CPF nem data e não comente a autenticação.\n"
        "Escolha exatamente uma ação fechada:\n"
        "- consult_limit: o cliente pediu para ver o limite. O sistema consulta "
        "e escreve o valor; você não digita o número.\n"
        "- ask_amount: o cliente pediu aumento sem valor em reais.\n"
        "- request_increase: a fala traz um valor em reais. Informe "
        "requested_limit em dígitos, sem R$ e sem ponto de milhar "
        "(R$ 8.000,00 vira 8000). O sistema grava e decide; você não digita "
        "status nem valor. Não peça confirmação.\n"
        "- handoff_interview: só depois que o cliente aceitar a entrevista "
        "oferecida após uma rejeição.\n"
        "- handoff_currency: o cliente quer cotação.\n"
        "- end: o cliente pediu para encerrar. A última resposta da entrevista, "
        "mesmo começando com não, não é pedido de encerramento.\n"
        "- refuse_other_data: o cliente pediu CPF, score ou dados de outra "
        "pessoa. O sistema recusa e segue na conta de quem está falando.\n"
        "- speak: desvio curto sem inventar limite, sem listar dado alheio.\n"
        "Um valor diferente de um pedido já feito é outro pedido. Nunca ofereça "
        "a entrevista por conta própria.\n"
        f"Pedido rejeitado em aberto: {open_request}."
    )
