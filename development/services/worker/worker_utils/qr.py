import qrcode
from PIL import Image # Necesaria para la manipulación de la imagen del QR
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.style import Style
import os # Para acceder a variables de entorno y limpiar pantalla
import sys # Para salir del script

# --- Configuración del script ---
# Se recomienda pasar la URL como una variable de entorno 'QR_URL'.
# Si no está definida, se usa esta URL por defecto.
DEFAULT_URL = "https://www.linkedin.com/in/wisrovi-rodriguez"
# --- Fin de la configuración del script ---

def generate_qr_ascii(url: str, box_size: int = 1, border: int = 1) -> str:
    """
    Genera un código QR de la URL dada y lo representa como una cadena de caracteres ASCII.
    """
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=box_size,
            border=border,
        )
        qr.add_data(url)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white").convert("1")
        width, height = img.size

        qr_ascii_lines = []
        for y in range(height):
            row_str = ""
            for x in range(width):
                if img.getpixel((x, y)) == 0:
                    row_str += "██"
                else:
                    row_str += "  "
            qr_ascii_lines.append(row_str)
        return "\n".join(qr_ascii_lines)
    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error al generar el QR ASCII:[/bold red] {e}", err=True)
        return ""

def show_qr_in_terminal():
    """
    Muestra un código QR en la terminal. La URL se obtiene de la variable de entorno QR_URL,
    o usa una URL por defecto si no está definida.
    """
    console = Console()

    # Limpia la pantalla de la terminal para una presentación limpia
    os.system('cls' if os.name == 'nt' else 'clear')

    # Intenta obtener la URL de la variable de entorno 'QR_URL'
    url_to_display = os.getenv('QR_URL', DEFAULT_URL)

    # Validación básica de formato de URL
    if not url_to_display.startswith(('http://', 'https://')):
        console.print("[bold red]ERROR: URL inválida. Debe comenzar con http:// o https://[/bold red]", err=True)
        sys.exit(1)

    # Genera el contenido ASCII del QR
    qr_ascii_content = generate_qr_ascii(url_to_display)

    if not qr_ascii_content:
        console.print("[bold red]ERROR: No se pudo generar el código QR.[/bold red]", err=True)
        sys.exit(1)

    # Contenido del panel
    panel_title = Text("Escanea para visitar la URL", style="bold green")
    panel_subtitle = Text(f"URL: {url_to_display}", style="dim cyan")
    panel_footer = Text("¡Gracias por visitar!", style="bold blue")

    panel_content = Panel(
        Align.center(Text(qr_ascii_content, style=Style(color="white", bgcolor="black"))),
        title=panel_title,
        subtitle=panel_subtitle,
        border_style="magenta",
        expand=False
    )

    # Imprime el panel y los mensajes adicionales
    console.print("\n")
    console.print(Align.center(panel_content))
    console.print("\n")
    console.print(Align.center(panel_footer))
    console.print("\n")

if __name__ == "__main__":
    show_qr_in_terminal()
