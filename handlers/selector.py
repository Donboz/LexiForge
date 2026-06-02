#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

def select_file_interactive(files, title="Select a file / Bir dosya seçin:"):
    """
    Interactive file selector using arrow keys on Windows, fallback to numbers on Unix.
    Windows üzerinde ok tuşlarıyla, Unix üzerinde numaralarla interaktif dosya seçici.
    """
    if not files:
        return None

    # Check for Windows virtual terminal support / Windows sanal terminal desteğini kontrol et
    is_windows = False
    try:
        import msvcrt
        is_windows = True
    except ImportError:
        pass

    # Enable Virtual Terminal Processing on Windows for ANSI support / ANSI desteği için Windows'ta Sanal Terminal İşlemeyi etkinleştir
    if is_windows:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h_stdout = kernel32.GetStdHandle(-11) # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h_stdout, ctypes.byref(mode)):
                # 0x0004 is ENABLE_VIRTUAL_TERMINAL_PROCESSING / 0x0004 ENABLE_VIRTUAL_TERMINAL_PROCESSING değeridir
                kernel32.SetConsoleMode(h_stdout, mode.value | 0x0004)
        except Exception:
            pass

    selected_index = 0
    num_files = len(files)

    print(f"\n\033[1;33m=== {title} ===\033[0m")
    print("\033[90m(Use UP/DOWN arrow keys to navigate, ENTER to select, ESC or Q to cancel) / "
          "(Gezinmek için YUKARI/AŞAĞI ok tuşlarını, seçmek için ENTER'ı, iptal için ESC veya Q'yu kullanın)\033[0m\n")

    # Hide cursor / İmleci gizle
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    def print_menu():
        for i, f in enumerate(files):
            basename = os.path.basename(f)
            if i == selected_index:
                # Cyan bold colored arrow and text for selected item / Seçilen öğe için turkuaz kalın ok işareti ve metin
                sys.stdout.write(f"\033[K\033[1;36m > {basename}\033[0m\n")
            else:
                sys.stdout.write(f"\033[K   {basename}\n")
        sys.stdout.flush()

    # Initial draw / İlk çizim
    print_menu()

    if is_windows:
        try:
            while True:
                ch = msvcrt.getch()
                if ch in (b'\xe0', b'\x00'): # Arrow key prefix / Ok tuşu öneki
                    ch2 = msvcrt.getch()
                    if ch2 == b'H': # Up Arrow / Yukarı Ok
                        selected_index = (selected_index - 1) % num_files
                    elif ch2 == b'P': # Down Arrow / Aşağı Ok
                        selected_index = (selected_index + 1) % num_files
                elif ch == b'\r': # Enter / Giriş
                    # Move cursor to end of list / İmleci listenin sonuna taşı ve göster
                    sys.stdout.write("\033[?25h") # Show cursor / İmleci göster
                    sys.stdout.flush()
                    return files[selected_index]
                elif ch in (b'\x1b', b'q', b'Q'): # Esc or Q / Esc veya Q
                    sys.stdout.write("\033[?25h") # Show cursor / İmleci göster
                    sys.stdout.flush()
                    return None

                # Reposition cursor to the start of the list / İmleci listenin başına geri taşı
                sys.stdout.write(f"\033[{num_files}A")
                print_menu()
        except KeyboardInterrupt:
            sys.stdout.write("\033[?25h") # Show cursor / İmleci göster
            sys.stdout.flush()
            return None
    else:
        # Fallback for non-Windows (Unix / macOS) / Windows dışı (Unix / macOS) için yedek yöntem
        sys.stdout.write("\033[?25h") # Show cursor / İmleci göster
        sys.stdout.flush()
        print("Interactive arrow selection is fully supported on Windows. For this environment, please select by number: / "
              "İnteraktif ok tuşu seçimi Windows üzerinde tam olarak desteklenmektedir. Bu ortam için lütfen numara ile seçin:")
        for i, f in enumerate(files):
            print(f"[{i+1}] {os.path.basename(f)}")
        try:
            choice = input(f"Select 1-{num_files} (or Q to cancel) / 1-{num_files} seçin (veya iptal için Q): ").strip()
            if choice.lower() in ('q', ''):
                return None
            idx = int(choice) - 1
            if 0 <= idx < num_files:
                return files[idx]
        except Exception:
            pass
        return None


def select_item_interactive(items, title="Select / Seçin:"):
    """
    Interactive list item selector using arrow keys on Windows, fallback to numbers on Unix.
    Windows üzerinde ok tuşlarıyla, Unix üzerinde numaralarla interaktif liste öğesi seçici.
    """
    if not items:
        return None

    is_windows = False
    try:
        import msvcrt
        is_windows = True
    except ImportError:
        pass

    if is_windows:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h_stdout = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h_stdout, ctypes.byref(mode)):
                kernel32.SetConsoleMode(h_stdout, mode.value | 0x0004)
        except Exception:
            pass

    selected_index = 0
    num_items = len(items)

    print(f"\n\033[1;33m=== {title} ===\033[0m")
    print("\033[90m(Use UP/DOWN arrow keys to navigate, ENTER to select, ESC or Q to cancel) / "
          "(Gezinmek için YUKARI/AŞAĞI ok tuşlarını, seçmek için ENTER'ı, iptal için ESC veya Q'yu kullanın)\033[0m\n")

    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    def print_menu():
        for i, item in enumerate(items):
            if i == selected_index:
                sys.stdout.write(f"\033[K\033[1;36m > {item}\033[0m\n")
            else:
                sys.stdout.write(f"\033[K   {item}\n")
        sys.stdout.flush()

    print_menu()

    if is_windows:
        try:
            while True:
                ch = msvcrt.getch()
                if ch in (b'\xe0', b'\x00'):
                    ch2 = msvcrt.getch()
                    if ch2 == b'H':
                        selected_index = (selected_index - 1) % num_items
                    elif ch2 == b'P':
                        selected_index = (selected_index + 1) % num_items
                elif ch == b'\r':
                    sys.stdout.write("\033[?25h")
                    sys.stdout.flush()
                    return items[selected_index]
                elif ch in (b'\x1b', b'q', b'Q'):
                    sys.stdout.write("\033[?25h")
                    sys.stdout.flush()
                    return None

                sys.stdout.write(f"\033[{num_items}A")
                print_menu()
        except KeyboardInterrupt:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            return None
    else:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        print("Interactive arrow selection is fully supported on Windows. For this environment, please select by number: / "
              "İnteraktif ok tuşu seçimi Windows üzerinde tam olarak desteklenmektedir. Bu ortam için lütfen numara ile seçin:")
        for i, item in enumerate(items):
            print(f"[{i+1}] {item}")
        try:
            choice = input(f"Select 1-{num_items} (or Q to cancel) / 1-{num_items} seçin (veya iptal için Q): ").strip()
            if choice.lower() in ('q', ''):
                return None
            idx = int(choice) - 1
            if 0 <= idx < num_items:
                return items[idx]
        except Exception:
            pass
        return None


def select_boolean_interactive(title="Confirm / Onaylayın:", default=True):
    """
    Bilingual Yes/No selector.
    Çift dilli Evet/Hayır seçici.
    """
    options = [
        "Yes (Recommended) / Evet (Önerilen)", 
        "No / Hayır"
    ] if default else [
        "No (Default) / Hayır (Varsayılan)", 
        "Yes / Evet"
    ]
    choice = select_item_interactive(options, title)
    if choice is None:
        return default
    return "Yes" in choice or "Evet" in choice
