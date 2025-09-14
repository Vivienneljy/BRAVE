from defending.FLAME import FLAME
from defending.BRAVE import BRAVE
from defending.LFighter import LFighter
from defending.XMAM import XMAM
from defending.DPFLA import DPFLA


def attack_detection_strategy(grad_list, args, device):
    if args.defense == 'brave':
        defense = BRAVE(grad_list, device, args)
    elif args.defense == 'flame':
        defense = FLAME(grad_list, device, args)
    elif args.defense == 'xmam':
        defense = XMAM(grad_list, device, args)
    elif args.defense == 'lfighter':
        defense = LFighter(grad_list, device, args)
    elif args.defense == 'dpfla':
        defense = DPFLA(grad_list, device, args)
    else:
        raise NotImplementedError
    return defense

