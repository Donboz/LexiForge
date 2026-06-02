#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import math
import orjson

# Prepend parent directory to sys.path to resolve imports from handlers/
# Üst dizini sys.path'e ekleyerek handlers/ modüllerini yüklemeyi sağlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from handlers.selector import select_item_interactive, select_file_interactive

def main():
    json_dir = os.path.join("data", "final")
    export_dir = os.path.join("data", "exports")

    # Ensure directories exist / Klasörlerin var olduğundan emin ol
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    # Find JSON files in data/final / data/final dizinindeki JSON dosyalarını bul
    json_files = []
    for f in os.listdir(json_dir):
        if f.endswith(".json") and not f.endswith(".tmp") and "sozluk_" in f and "_progress" not in f:
            json_files.append(os.path.join(json_dir, f))

    if not json_files:
        print(f"\033[1;31mError: No JSON dictionary files to split found in '{json_dir}'. / "
              f"Hata: '{json_dir}' dizininde bölünecek JSON sözlük dosyası bulunamadı.\033[0m")
        sys.exit(1)

    # Interactive choice: split single or all? / İnteraktif seçim: tek bir dosya mı yoksa hepsi mi?
    action_options = [
        "Split a single JSON file / Tek bir JSON dosyasını böl", 
        "Split all JSON files / Tüm JSON dosyalarını böl"
    ]
    selected_action = select_item_interactive(action_options, "Please select an action / Lütfen yapmak istediğiniz işlemi seçin:")
    if not selected_action:
        print("Action cancelled. / İşlem iptal edildi.")
        sys.exit(0)

    files_to_process = []
    if "Split all" in selected_action or "Tüm JSON" in selected_action:
        files_to_process = json_files
    else:
        selected_file = select_file_interactive(json_files, "Select the JSON dictionary file to split / Bölmek istediğiniz JSON sözlük dosyasını seçin")
        if not selected_file:
            print("Action cancelled. / İşlem iptal edildi.")
            sys.exit(0)
        files_to_process.append(selected_file)

    # Chunk size selection / Parça boyutu seçimi
    chunk_options = [
        "5000 (Recommended - Safe Import) / 5000 (Önerilen - Güvenli İçe Aktarım)", 
        "1000", 
        "10000", 
        "Enter custom value... / Özel Değer Girin..."
    ]
    selected_chunk = select_item_interactive(chunk_options, "What should be the chunk size? / Bölme boyutu (Chunk Size) ne olsun?")
    chunk_size = 5000
    if not selected_chunk or "5000" in selected_chunk:
        chunk_size = 5000
    elif selected_chunk == "1000":
        chunk_size = 1000
    elif selected_chunk == "10000":
        chunk_size = 10000
    else:
        try:
            val = input("Enter custom chunk size (e.g. 2500) / Özel bölme boyutu girin (örn. 2500): ").strip()
            chunk_size = int(val) if val else 5000
        except ValueError:
            chunk_size = 5000
            print("Invalid value. Defaulting to 5000. / Geçersiz değer. Varsayılan olarak 5000 seçildi.")

    print(f"\n\033[1;36mStarting split process... (Chunk Size: {chunk_size}) / "
          f"Bölme işlemi başlatılıyor... (Chunk Size: {chunk_size})\033[0m\n")

    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        base_name = file_name.replace(".json", "")
        print(f"Processing / İşleniyor: {file_name}")

        try:
            with open(file_path, "rb") as f:
                terms = orjson.loads(f.read())
            
            if not isinstance(terms, list):
                print(f"  \033[1;31mSkipped: {file_name} does not contain a valid list. / Atlandı: {file_name} geçerli bir liste içermiyor.\033[0m")
                continue

            total_terms = len(terms)
            if total_terms == 0:
                print(f"  \033[1;33mSkipped: {file_name} is empty. / Atlandı: {file_name} boş.\033[0m")
                continue

            num_chunks = math.ceil(total_terms / chunk_size)
            print(f"  Total terms: {total_terms}. Will split into {num_chunks} chunks. / Toplam terim sayısı: {total_terms}. Toplam {num_chunks} parçaya bölünecek.")

            for i in range(num_chunks):
                chunk_data = terms[i * chunk_size : (i + 1) * chunk_size]
                chunk_num = i + 1
                chunk_file_name = f"{base_name}_chunk_{chunk_num}.json"
                chunk_file_path = os.path.join(export_dir, chunk_file_name)

                with open(chunk_file_path, "wb") as out_f:
                    out_f.write(orjson.dumps(chunk_data, option=orjson.OPT_INDENT_2))

                print(f"    -> Saved / Kaydedildi: data/exports/{chunk_file_name} ({len(chunk_data)} terms/terim)")

            print(f"  \033[1;32mSuccessfully completed / Başarıyla tamamlandı: {file_name}\033[0m\n")

        except Exception as e:
            print(f"  \033[1;31mError occurred / Hata oluştu ({file_name}): {e}\033[0m\n")

    print("\033[1;32mAll splits completed. Outputs saved to 'data/exports/'. / "
          "Tüm bölme işlemleri tamamlandı. Çıktılar 'data/exports/' klasörüne kaydedildi.\033[0m")

if __name__ == "__main__":
    main()
