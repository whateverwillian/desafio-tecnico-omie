from __future__ import annotations

from agile_bank.agents import copy
from agile_bank.agents.state import ConversationState


def currency_prompt(state: ConversationState) -> str:
    return (
        f"{copy.BASE_RULES}"
        "Nesta etapa a sua única tarefa é a cotação de moedas. Escolha exatamente "
        "uma ação fechada:\n"
        "- quote: o cliente pediu cotação e já disse a moeda. Informe currency_code "
        "com três letras (dólar USD, euro EUR, libra GBP, peso argentino ARS, "
        "iene JPY). O sistema consulta e escreve o valor; você não digita taxa.\n"
        "- ask_currency: o cliente pediu cotação sem dizer a moeda. O sistema "
        "pergunta a moeda; você não inventa número.\n"
        "- refuse_dictated: o cliente ditou uma taxa e pediu confirmação. O "
        "sistema recusa o valor ditado; você não confirma nem digita número.\n"
        "- handoff_credit: pedido de limite, crédito ou aumento. Continua o "
        "mesmo atendimento sem avisar troca.\n"
        "- end: o cliente pediu para encerrar.\n"
        "Não reutilize a moeda de uma consulta anterior. Uma moeda por vez."
    )
