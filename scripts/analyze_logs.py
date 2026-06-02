#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import glob
import json
from datetime import datetime
from collections import defaultdict

# Prepend parent directory to sys.path to resolve imports from handlers/
# Üst dizini sys.path'e ekleyerek handlers/ modüllerini yüklemeyi sağlar
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ANSI styling helper / ANSI stil yardımcı sınıfı
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def parse_ts(ts_str):
    """Robust ISO 8601 timestamp parser / Güvenilir ISO 8601 zaman damgası ayrıştırıcı."""
    try:
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + '+00:00'
        return datetime.fromisoformat(ts_str)
    except Exception:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(ts_str, fmt)
            except Exception:
                continue
        return None

def get_all_configured_models():
    """Extracts all configured models from config files / Ayarlardan tüm modelleri çeker."""
    models_set = set()
    
    config_path = os.path.join("config", "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                api_providers = cfg.get("api_providers", {})
                for provider, pcfg in api_providers.items():
                    for m in pcfg.get("models", []):
                        models_set.add((provider, m))
        except Exception:
            pass
            
    return models_set

def make_bar(percentage, width=15):
    """Draws a visual progress bar representing success rate / Başarı oranını gösteren bir bar çizer."""
    filled = int(round((percentage / 100.0) * width))
    empty = width - filled
    bar_str = f"{Colors.GREEN}{'█' * filled}{Colors.RED}{'░' * empty}{Colors.RESET}"
    return bar_str

def main():
    log_dir = os.path.join("data", "logs")
    log_files = glob.glob(os.path.join(log_dir, "*.jsonl"))
    
    configured_models = get_all_configured_models()
    stats = {}
    
    total_parsed_lines = 0
    total_corrupt_lines = 0
    
    for log_file in log_files:
        pending_attempts = {}
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    total_parsed_lines += 1
                    try:
                        row = json.loads(line)
                    except Exception:
                        total_corrupt_lines += 1
                        continue
                        
                    event = row.get("event")
                    ts_str = row.get("ts")
                    data = row.get("data", {})
                    if not event or not ts_str:
                        continue
                        
                    provider = data.get("provider")
                    model = data.get("model")
                    if not provider or not model:
                        continue
                        
                    ts = parse_ts(ts_str)
                    if not ts:
                        continue
                        
                    key = (provider, model)
                    
                    if key not in stats:
                        stats[key] = {
                            "attempts": 0,
                            "successes": 0,
                            "failures": 0,
                            "rate_limits": 0,
                            "latencies": []
                        }
                    
                    if event == "fallback_model_attempt":
                        if key not in pending_attempts:
                            pending_attempts[key] = []
                        pending_attempts[key].append(ts)
                        stats[key]["attempts"] += 1
                        
                    elif event == "fallback_model_success":
                        stats[key]["successes"] += 1
                        if key in pending_attempts and pending_attempts[key]:
                            start_ts = pending_attempts[key].pop(0)
                            latency = (ts - start_ts).total_seconds()
                            if latency >= 0:
                                stats[key]["latencies"].append(latency)
                                
                    elif event == "fallback_model_failed":
                        stats[key]["failures"] += 1
                        if key in pending_attempts and pending_attempts[key]:
                            start_ts = pending_attempts[key].pop(0)
                            latency = (ts - start_ts).total_seconds()
                            if latency >= 0:
                                stats[key]["latencies"].append(latency)
                                
                    elif event in ("fallback_model_rate_limited", "fallback_model_exhausted"):
                        stats[key]["rate_limits"] += 1
                        if key in pending_attempts and pending_attempts[key]:
                            start_ts = pending_attempts[key].pop(0)
                            latency = (ts - start_ts).total_seconds()
                            if latency >= 0:
                                stats[key]["latencies"].append(latency)
        except Exception as e:
            print(f"Error reading file / Dosya okunurken hata oluştu {log_file}: {e}")

    processed_stats = []
    
    total_attempts = 0
    total_successes = 0
    total_failures = 0
    total_rate_limits = 0
    all_latencies = []
    
    for key, val in stats.items():
        provider, model = key
        outcomes_sum = val["successes"] + val["failures"] + val["rate_limits"]
        attempts = max(val["attempts"], outcomes_sum)
        
        successes = val["successes"]
        failures = val["failures"]
        rate_limits = val["rate_limits"]
        
        success_rate = (successes / outcomes_sum * 100) if outcomes_sum > 0 else 0.0
        
        latencies = val["latencies"]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        min_latency = min(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0
        
        total_attempts += attempts
        total_successes += successes
        total_failures += failures
        total_rate_limits += rate_limits
        all_latencies.extend(latencies)
        
        processed_stats.append({
            "provider": provider,
            "model": model,
            "attempts": attempts,
            "successes": successes,
            "failures": failures,
            "rate_limits": rate_limits,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "latencies_count": len(latencies)
        })

    active_models = set(stats.keys())
    never_ran = sorted(list(configured_models - active_models), key=lambda x: (x[0], x[1]))

    # Print Dashboard
    print(f"\n{Colors.BOLD}{Colors.CYAN}┌──────────────────────────────────────────────────────────┐")
    print(f"│            GLOSSA LLM PERFORMANCE DASHBOARD              │")
    print(f"│            GLOSSA LLM PERFORMANS GÖSTERGESİ              │")
    print(f"└──────────────────────────────────────────────────────────┘{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}=== GLOBAL METRICS / GENEL METRİKLER ==={Colors.RESET}")
    print(f"  Log Files Parsed / Çözümlenen Log Dosyası : {Colors.BOLD}{len(log_files)}{Colors.RESET}")
    print(f"  Total API Calls / Toplam API Çağrısı      : {Colors.BOLD}{total_attempts}{Colors.RESET}")
    print(f"  Successes / Başarılı Deneme               : {Colors.GREEN}{Colors.BOLD}{total_successes}{Colors.RESET}")
    print(f"  Failures / Hatalı Deneme                  : {Colors.RED}{Colors.BOLD}{total_failures}{Colors.RESET}")
    print(f"  Rate Limits / Rate Limit Engeli           : {Colors.YELLOW}{Colors.BOLD}{total_rate_limits}{Colors.RESET}")
    
    global_success_rate = (total_successes / (total_successes + total_failures + total_rate_limits) * 100) if (total_successes + total_failures + total_rate_limits) > 0 else 0
    print(f"  Global Success Rate / Başarı Oranı        : {Colors.BOLD}{global_success_rate:.2f}%{Colors.RESET} {make_bar(global_success_rate, width=20)}")
    
    global_avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    print(f"  Global Avg Latency / Ortalama Gecikme     : {Colors.BOLD}{global_avg_latency:.2f}s{Colors.RESET}")

    valid_latencies = [s for s in processed_stats if s["avg_latency"] > 0]
    fastest_model = min(valid_latencies, key=lambda x: x["avg_latency"]) if valid_latencies else None
    slowest_model = max(valid_latencies, key=lambda x: x["avg_latency"]) if valid_latencies else None
    
    ranked_models = sorted(processed_stats, key=lambda x: (-x["success_rate"], -x["successes"], x["avg_latency"]))
    
    provider_groups = defaultdict(lambda: {"successes": 0, "attempts": 0, "failures": 0, "rate_limits": 0, "latencies": []})
    for s in processed_stats:
        p = s["provider"]
        provider_groups[p]["successes"] += s["successes"]
        provider_groups[p]["attempts"] += s["attempts"]
        provider_groups[p]["failures"] += s["failures"]
        provider_groups[p]["rate_limits"] += s["rate_limits"]
        provider_groups[p]["latencies"].extend(stats[(s["provider"], s["model"])]["latencies"])
        
    provider_ranks = []
    for p, p_stats in provider_groups.items():
        total_p_outcomes = p_stats["successes"] + p_stats["failures"] + p_stats["rate_limits"]
        p_success_rate = (p_stats["successes"] / total_p_outcomes * 100) if total_p_outcomes > 0 else 0.0
        p_avg_latency = sum(p_stats["latencies"]) / len(p_stats["latencies"]) if p_stats["latencies"] else 0.0
        provider_ranks.append({
            "provider": p,
            "successes": p_stats["successes"],
            "attempts": p_stats["attempts"],
            "failures": p_stats["failures"],
            "rate_limits": p_stats["rate_limits"],
            "success_rate": p_success_rate,
            "avg_latency": p_avg_latency
        })
    provider_ranks = sorted(provider_ranks, key=lambda x: (-x["success_rate"], -x["successes"], x["avg_latency"]))

    print(f"\n{Colors.BOLD}{Colors.BLUE}=== MODEL PERFORMANCE RANKINGS / MODEL PERFORMANS SIRALAMASI ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GRAY}┌──────────────────────┬──────────────────────────────────────────┬───────────┬──────────────┬──────────────┬──────────────┬────────────┐")
    print(f"│ Provider             │ Model                                    │ Attempts  │ Success Rate │ Failures     │ Rate Limits  │ Avg Latency│")
    print(f"├──────────────────────┼──────────────────────────────────────────┼───────────┼──────────────┼──────────────┼──────────────┼────────────┤{Colors.RESET}")
    
    for s in ranked_models:
        prov = s["provider"][:20].ljust(20)
        mod = s["model"][:40].ljust(40)
        att = str(s["attempts"]).rjust(9)
        succ_rate = f"{s['success_rate']:.1f}%".rjust(11)
        fails = str(s["failures"]).rjust(12)
        rls = str(s["rate_limits"]).rjust(12)
        lat = f"{s['avg_latency']:.2f}s".rjust(10) if s['avg_latency'] > 0 else "N/A".rjust(10)
        
        if s['success_rate'] >= 90:
            rate_color = Colors.GREEN
        elif s['success_rate'] >= 60:
            rate_color = Colors.YELLOW
        else:
            rate_color = Colors.RED
            
        print(f"│ {prov} │ {mod} │ {att} │ {rate_color}{succ_rate}{Colors.RESET} │ {fails} │ {rls} │ {lat} │")
        
    print(f"{Colors.BOLD}{Colors.GRAY}└──────────────────────┴──────────────────────────────────────────┴───────────┴──────────────┴──────────────┴──────────────┴────────────┘{Colors.RESET}")

    # Provider Summary Table
    print(f"\n{Colors.BOLD}{Colors.BLUE}=== PROVIDER PERFORMANCE SUMMARY / SAĞLAYICI PERFORMANS ÖZETİ ==={Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GRAY}┌──────────────────────┬───────────┬──────────────┬──────────────┬──────────────┬────────────┐")
    print(f"│ Provider             │ Attempts  │ Success Rate │ Failures     │ Rate Limits  │ Avg Latency│")
    print(f"├──────────────────────┼───────────┼──────────────┼──────────────┼──────────────┼────────────┤{Colors.RESET}")
    
    for pr in provider_ranks:
        prov = pr["provider"][:20].ljust(20)
        att = str(pr["attempts"]).rjust(9)
        succ_rate = f"{pr['success_rate']:.1f}%".rjust(11)
        fails = str(pr["failures"]).rjust(12)
        rls = str(pr["rate_limits"]).rjust(12)
        lat = f"{pr['avg_latency']:.2f}s".rjust(10) if pr['avg_latency'] > 0 else "N/A".rjust(10)
        
        if pr['success_rate'] >= 90:
            rate_color = Colors.GREEN
        elif pr['success_rate'] >= 60:
            rate_color = Colors.YELLOW
        else:
            rate_color = Colors.RED
            
        print(f"│ {prov} │ {att} │ {rate_color}{succ_rate}{Colors.RESET} │ {fails} │ {rls} │ {lat} │")
    print(f"{Colors.BOLD}{Colors.GRAY}└──────────────────────┴───────────┴──────────────┴──────────────┴──────────────┴────────────┘{Colors.RESET}")

    # Key Highlights
    print(f"\n{Colors.BOLD}{Colors.BLUE}=== KEY PERFORMANCE HIGHLIGHTS / ÖNEMLİ BULGULAR ==={Colors.RESET}")
    if ranked_models:
        best_model = ranked_models[0]
        worst_model = sorted(processed_stats, key=lambda x: (x["success_rate"], -x["failures"], -x["avg_latency"]))[0]
        
        print(f"  🏆 {Colors.BOLD}Best Model / En Güvenilir Model {Colors.RESET} : {Colors.GREEN}{best_model['provider']}/{best_model['model']}{Colors.RESET} ({best_model['success_rate']:.1f}% Success, {best_model['attempts']} calls)")
        if fastest_model:
            print(f"  ⚡ {Colors.BOLD}Fastest Response / En Hızlı    {Colors.RESET} : {Colors.GREEN}{fastest_model['provider']}/{fastest_model['model']}{Colors.RESET} ({fastest_model['avg_latency']:.2f}s avg)")
        if slowest_model:
            print(f"  🐢 {Colors.BOLD}Slowest Response / En Yavaş    {Colors.RESET} : {Colors.RED}{slowest_model['provider']}/{slowest_model['model']}{Colors.RESET} ({slowest_model['avg_latency']:.2f}s avg)")
        print(f"  ⚠️  {Colors.BOLD}Worst Model / En İstikrarsız   {Colors.RESET} : {Colors.RED}{worst_model['provider']}/{worst_model['model']}{Colors.RESET} ({worst_model['success_rate']:.1f}% Success, {worst_model['attempts']} calls)")
        
        best_prov = provider_ranks[0]
        print(f"  🏢 {Colors.BOLD}Best Provider / En Başarılı API{Colors.RESET} : {Colors.GREEN}{best_prov['provider']}{Colors.RESET} ({best_prov['success_rate']:.1f}% Success Rate)")
    else:
        print(f"  No active model statistics found in log files. / Log dosyalarında aktif model istatistiği bulunamadı.")

    # Never Run Models
    print(f"\n{Colors.BOLD}{Colors.BLUE}=== MODELS THAT NEVER RAN (0 LOGGED ATTEMPTS) / HİÇ ÇALIŞMAYAN MODELLER (0 LOGLANAN DENEME) ==={Colors.RESET}")
    if never_ran:
        current_provider = None
        provider_models = []
        for prov, mod in never_ran:
            if prov != current_provider:
                if current_provider:
                    print(f"  • {Colors.BOLD}{current_provider}{Colors.RESET}: {Colors.GRAY}{', '.join(provider_models)}{Colors.RESET}")
                current_provider = prov
                provider_models = [mod]
            else:
                provider_models.append(mod)
        if current_provider:
            print(f"  • {Colors.BOLD}{current_provider}{Colors.RESET}: {Colors.GRAY}{', '.join(provider_models)}{Colors.RESET}")
        print(f"\n  Total Unused Models / Toplam Kullanılmayan Model: {Colors.BOLD}{len(never_ran)}{Colors.RESET}")
    else:
        print("  All configured models have been executed at least once. / Ayarlanmış tüm modeller en az bir kere çalıştırıldı.")
        
    print()

if __name__ == "__main__":
    main()
