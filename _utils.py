import datetime

def format_time(elapsed: float) -> str:
    """Format seconds to hh:mm:ss."""
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))
