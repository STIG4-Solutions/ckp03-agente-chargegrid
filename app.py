"""
Executa a interface com URL pública: `python app.py`.

O link público (`share=True`) fica ativo enquanto este processo estiver
rodando. Para a avaliação, rode numa máquina que fique ligada; no macOS,
`caffeinate -dimsu python app.py` impede que o computador durma.
Para testar só na máquina local: `APP_SHARE=0 python app.py`.
"""

import os

from ckp03.agente import AgenteChargeGrid
from ckp03.interface import construir_interface

if __name__ == "__main__":
    demo = construir_interface(AgenteChargeGrid(verbose=True))
    demo.queue(default_concurrency_limit=4).launch(
        share=os.getenv("APP_SHARE", "1") != "0",
        server_port=int(os.getenv("APP_PORT", "7860")),
        theme="soft",
    )
