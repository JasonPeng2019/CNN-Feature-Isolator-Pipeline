"""Train and freeze the CIFAR ResNet backbone (Phase 0)."""
import argparse, os, sys, time, json
import torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import data as D
import backbone as B


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet56")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="./runs/backbone")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = args.device

    train_loader, test_loader = D.get_loaders(args.dataset, root="./data", batch_size=args.bs)
    nc = D.num_classes(args.dataset)
    model = {"resnet20": B.resnet20, "resnet56": B.resnet56, "resnet110": B.resnet110}[args.arch](nc).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    log_path = os.path.join(args.out, "train.log")
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.size(0)
        sched.step()
        acc = evaluate(model, test_loader, device)
        best = max(best, acc)
        if acc >= best:
            torch.save({"model": model.state_dict(), "arch": args.arch,
                        "dataset": args.dataset, "acc": acc}, os.path.join(args.out, "best.pt"))
        msg = f"ep {ep+1}/{args.epochs} loss {run_loss/len(train_loader.dataset):.4f} test_acc {acc:.2f} best {best:.2f} ({time.time()-t0:.1f}s)"
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    with open(os.path.join(args.out, "done.json"), "w") as f:
        json.dump({"best_acc": best, "arch": args.arch, "dataset": args.dataset}, f)
    print(f"DONE best_acc={best:.2f}", flush=True)


if __name__ == "__main__":
    main()
