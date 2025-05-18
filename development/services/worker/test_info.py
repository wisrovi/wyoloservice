from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

# Datos del log parseados manualmente (puedes automatizar esto si lees desde archivo)
info_data = {
    "__VERSION__": "v1.0.10",
    "Results queue": "results_queue",
    "Debug mode": "None"
}

topics_list = [
    "admin_192.168.1.137",
    "stop_192.168.1.137",
    "wisrovi",
    "192.168.1.137",
    "training_queue"
]

# Tabla para la información principal
table_info = Table(title="Información Inicial", show_header=True, header_style="bold magenta")
table_info.add_column("Campo", style="cyan")
table_info.add_column("Valor", style="green")

for key, value in info_data.items():
    table_info.add_row(key, value)

# Panel con los tópicos como lista
topics_text = Text("\n".join(f"• {topic}" for topic in topics_list), style="yellow")
panel_topics = Panel(topics_text, title="Tópicos Activados", border_style="blue")

# Mostrar todo en consola
console.print(table_info)
console.print(panel_topics)
console.print("[bold green]✅ Health check started with version v1.0.10[/bold green]")