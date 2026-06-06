#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LexiForge Interactive CLI Selector
==================================
Handles interactive console selections using arrow keys on Windows, fallback to numbers on Unix.
Supports color-coded completion status badges and blocking of completed/finished options.
"""

import os
import sys

def select_file_interactive(files, title="Select a file / Bir dosya seçin:", status_dict=None):
    """
    Interactive file selector using arrow keys on Windows, fallback to numbers on Unix.
    Renders completed files as dimmed and blocks their selection.
    """
    if not files:
        return None

    # Check for Windows virtual terminal support
    is_windows = False
    try:
        import msvcrt
        is_windows = True
    except ImportError:
        pass

    # Enable Virtual Terminal Processing on Windows for ANSI support
    if is_windows:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h_stdout = kernel32.GetStdHandle(-11) # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h_stdout, ctypes.byref(mode)):
                kernel32.SetConsoleMode(h_stdout, mode.value | 0x0004)
        except Exception:
            pass

    selected_index = 0
    num_files = len(files)
    status_dict = status_dict or {}

    print(f"\n\033[1;33m=== {title} ===\033[0m")
    print("\033[90m(Use UP/DOWN arrow keys to navigate, ENTER to select, ESC or Q to cancel) / "
          "(Gezinmek için YUKARI/AŞAĞI ok tuşlarını, seçmek için ENTER'ı, iptal için ESC veya Q'yu kullanın)\033[0m\n")

    # Hide cursor
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    # We will keep a placeholder line at the bottom for warning messages
    # Hata/uyarı mesajları için en altta bir satır boşluk bırakırız
    sys.stdout.write("\n")
    sys.stdout.flush()

    def print_menu(warning_msg=""):
        # Move up to start of menu (num_files lines + 1 warning line + 1 extra newline)
        # Menünün başına gitmek için imleci yukarı kaydırırız
        sys.stdout.write(f"\033[{num_files + 1}A")
        
        for i, f in enumerate(files):
            basename = os.path.basename(f)
            status = status_dict.get(f)
            
            if status == "COMPLETE":
                if i == selected_index:
                    # Selected but completed (blocked) / Seçili ama tamamlanmış (kilitli)
                    sys.stdout.write(f"\033[K\033[90m > {basename}\033[0m \033[1;30;42m DONE / TAMAMLANDI \033[0m\n")
                else:
                    # Unselected and completed / Seçili değil ve tamamlanmış
                    sys.stdout.write(f"\033[K\033[90m   {basename} [Finished / Tamamlandı]\033[0m\n")
            else:
                if i == selected_index:
                    # Selected and active / Seçili ve aktif
                    sys.stdout.write(f"\033[K\033[1;36m > {basename}\033[0m\n")
                else:
                    # Unselected and active / Seçili değil ve aktif
                    sys.stdout.write(f"\033[K   {basename}\n")
                    
        # Print warning/status message at the bottom line
        # En alttaki uyarı satırını güncelleriz
        if warning_msg:
            sys.stdout.write(f"\033[K\033[1;31m{warning_msg}\033[0m\n")
        else:
            sys.stdout.write("\033[K\n")
            
        sys.stdout.flush()

    # Initial draw
    # Menüyü ilk kez çizerken yukarı kaydırma yapmamak için dummy boşluklar bırakıp öyle çizeriz
    for _ in range(num_files + 1):
        sys.stdout.write("\n")
    print_menu()

    if is_windows:
        try:
            warning = ""
            while True:
                ch = msvcrt.getch()
                if ch in (b'\xe0', b'\x00'): # Arrow key prefix
                    ch2 = msvcrt.getch()
                    warning = "" # Clear warning on navigation
                    if ch2 == b'H': # Up Arrow
                        selected_index = (selected_index - 1) % num_files
                    elif ch2 == b'P': # Down Arrow
                        selected_index = (selected_index + 1) % num_files
                elif ch == b'\r': # Enter
                    selected_file = files[selected_index]
                    if status_dict.get(selected_file) == "COMPLETE":
                        warning = "⚠ [Blocked] File is already completed! / [Kilitli] Dosya zaten tamamlanmış!"
                        print_menu(warning)
                        continue
                        
                    sys.stdout.write("\033[?25h") # Show cursor
                    sys.stdout.flush()
                    return selected_file
                elif ch in (b'\x1b', b'q', b'Q'): # Esc or Q
                    sys.stdout.write("\033[?25h") # Show cursor
                    sys.stdout.flush()
                    return None

                print_menu(warning)
        except KeyboardInterrupt:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            return None
    else:
        # Fallback for non-Windows (Unix / macOS)
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        print("Interactive arrow selection is fully supported on Windows. Select by number: / Numara ile seçin:")
        
        selectable_indices = []
        for i, f in enumerate(files):
            status = status_dict.get(f)
            if status == "COMPLETE":
                print(f"\033[90m[{i+1}] {os.path.basename(f)} [Finished / Tamamlandı]\033[0m")
            else:
                print(f"[{i+1}] {os.path.basename(f)}")
                selectable_indices.append(i + 1)
                
        try:
            choice = input(f"Select 1-{num_files} (or Q to cancel): ").strip()
            if choice.lower() in ('q', ''):
                return None
            idx = int(choice) - 1
            if 0 <= idx < num_files:
                selected_file = files[idx]
                if status_dict.get(selected_file) == "COMPLETE":
                    print("\033[1;31m[Error] That file is completed and blocked! / Bu dosya tamamlanmış ve seçilemez!\033[0m")
                    return None
                return selected_file
        except Exception:
            pass
        return None


def select_item_interactive(items, title="Select / Seçin:", status_dict=None):
    """
    Interactive list item selector using arrow keys on Windows, fallback to numbers on Unix.
    Renders completed items as dimmed and blocks their selection.
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
    status_dict = status_dict or {}

    print(f"\n\033[1;33m=== {title} ===\033[0m")
    print("\033[90m(Use UP/DOWN arrow keys to navigate, ENTER to select, ESC or Q to cancel) / "
          "(Gezinmek için YUKARI/AŞAĞI ok tuşlarını, seçmek için ENTER'ı, iptal için ESC veya Q'yu kullanın)\033[0m\n")

    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    def print_menu(warning_msg=""):
        sys.stdout.write(f"\033[{num_items + 1}A")
        for i, item in enumerate(items):
            # Resolve key to lookup in status_dict (e.g. "EN - İngilizce" -> "EN")
            status_key = item.split(" - ")[0].strip() if " - " in str(item) else str(item)
            status = status_dict.get(status_key) or status_dict.get(str(item))
            
            if status == "COMPLETE":
                if i == selected_index:
                    sys.stdout.write(f"\033[K\033[90m > {item}\033[0m \033[1;30;42m DONE / TAMAMLANDI \033[0m\n")
                else:
                    sys.stdout.write(f"\033[K\033[90m   {item} [Finished / Tamamlandı]\033[0m\n")
            else:
                if i == selected_index:
                    sys.stdout.write(f"\033[K\033[1;36m > {item}\033[0m\n")
                else:
                    sys.stdout.write(f"\033[K   {item}\n")
                    
        if warning_msg:
            sys.stdout.write(f"\033[K\033[1;31m{warning_msg}\033[0m\n")
        else:
            sys.stdout.write("\033[K\n")
        sys.stdout.flush()

    for _ in range(num_items + 1):
        sys.stdout.write("\n")
    print_menu()

    if is_windows:
        try:
            warning = ""
            while True:
                ch = msvcrt.getch()
                if ch in (b'\xe0', b'\x00'):
                    ch2 = msvcrt.getch()
                    warning = ""
                    if ch2 == b'H':
                        selected_index = (selected_index - 1) % num_items
                    elif ch2 == b'P':
                        selected_index = (selected_index + 1) % num_items
                elif ch == b'\r':
                    selected_item = items[selected_index]
                    status_key = selected_item.split(" - ")[0].strip() if " - " in str(selected_item) else str(selected_item)
                    status = status_dict.get(status_key) or status_dict.get(str(selected_item))
                    
                    if status == "COMPLETE":
                        warning = "⚠ [Blocked] Option is already completed! / [Kilitli] Seçenek zaten tamamlanmış!"
                        print_menu(warning)
                        continue
                        
                    sys.stdout.write("\033[?25h")
                    sys.stdout.flush()
                    return selected_item
                elif ch in (b'\x1b', b'q', b'Q'):
                    sys.stdout.write("\033[?25h")
                    sys.stdout.flush()
                    return None

                print_menu(warning)
        except KeyboardInterrupt:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            return None
    else:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        print("Interactive arrow selection is fully supported on Windows. Select by number: / Numara ile seçin:")
        
        for i, item in enumerate(items):
            status_key = item.split(" - ")[0].strip() if " - " in str(item) else str(item)
            status = status_dict.get(status_key) or status_dict.get(str(item))
            if status == "COMPLETE":
                print(f"\033[90m[{i+1}] {item} [Finished / Tamamlandı]\033[0m")
            else:
                print(f"[{i+1}] {item}")
                
        try:
            choice = input(f"Select 1-{num_items} (or Q to cancel): ").strip()
            if choice.lower() in ('q', ''):
                return None
            idx = int(choice) - 1
            if 0 <= idx < num_items:
                selected_item = items[idx]
                status_key = selected_item.split(" - ")[0].strip() if " - " in str(selected_item) else str(selected_item)
                status = status_dict.get(status_key) or status_dict.get(str(selected_item))
                if status == "COMPLETE":
                    print("\033[1;31m[Error] That option is completed and blocked! / Bu seçenek tamamlanmış ve seçilemez!\033[0m")
                    return None
                return selected_item
        except Exception:
            pass
        return None


def select_boolean_interactive(title="Confirm / Onaylayın:", default=True):
    """
    Bilingual Yes/No selector.
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
