"""Train at 416 once, then test the same best.pt at 416 and 640."""
from size_experiment_common import main

if __name__ == '__main__':
    main('inference')
