import os
from datetime import date, datetime
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO
from util.logtool.logfilter import LogFilter
from util.logtool.clifilter import CLIFilter
from util.parser.mainparser import MainParser
from util.parser.cfgloader import ConfigLoader


class Recorder(object):
    def __init__(self):
        parser = MainParser()
        # コマンド受け取り
        self.args = parser.parse_args()
        self.cfg = ConfigLoader(self.args.cfg_path)

        # "output"ディレクトリがあるか確認
        assert os.path.isdir(self.cfg.GetPath("output")), f"\n[ERROR] incorrect dir: {self.cfg.GetPath('output')}"

        # プログラム実行日時を出力用ディレクトリ名にする
        d = datetime.now().strftime("%Y-%m%d-%H%M%S")
        self.cfg.cfg["path"]["output"] += d + "/"
        dir = self.cfg.GetPath("output")
        file = "output.log"
        # 存在しなければディレクトリを作成
        if not os.path.exists(dir):
            os.mkdir(dir)
        # ログ用ファイルを作成
        with open(os.path.join(dir, file), "w") as f:
            d = date.today().strftime("%Y-%m-%d")
            f.write(f"[Running Date : {d}]\n\n")

        # logger
        self.main_logger = getLogger("log_main")
        # 日時フォーマット
        datefmt = "%m-%d %H:%M:%S"
        # ファイル出力のフォーマット
        filefmt = "[%(levelname)-6s | %(asctime)s] %(message)s"
        file_formatter = Formatter(filefmt, datefmt)
        # 標準出力のフォーマット
        clifmt = "\033[%(n)dA[%(asctime)s] %(message)s\n"
        cli_formatter = Formatter(clifmt, datefmt)
        # ファイル出力のハンドラ
        file_handler = FileHandler(os.path.join(dir, file), mode="a+", encoding="utf_8")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(DEBUG)
        # ファイル出力のフィルタ
        file_filter = LogFilter()
        file_handler.addFilter(file_filter)
        # 標準出力のハンドラ
        cli_handler = StreamHandler()
        cli_handler.terminator = ""
        cli_handler.setFormatter(cli_formatter)
        cli_handler.setLevel(INFO)
        # 標準出力のフィルタ
        cli_filter = CLIFilter()
        cli_handler.addFilter(cli_filter)
        # loggerを設定
        self.main_logger.addHandler(cli_handler)
        self.main_logger.addHandler(file_handler)
        self.main_logger.setLevel(DEBUG)

    def Display(self, *args):
        status = ""

        for arg in args:
            status += arg + "\n"

        self.Info("",
                  extra={"status": {"stat": status, 
                                    "n": len(args)+1}})

    def Debug(self, msg):
        self.main_logger.debug(msg)
    
    def Info(self, msg, extra={"n": 0}):
        self.main_logger.info(msg, extra=extra)

    def Warning(self, msg):
        self.main_logger.warning(msg=msg)

    def Error(self, msg):
        self.main_logger.error(msg=msg)

    def Critical(self, msg):
        self.main_logger.critical(msg=msg)

    def SetLogDigits(self, epoch, iteration):
        if not self.main_logger.hasHandlers():
            self.Debug("[NOTICE] SetLogDigits failed. main_logger has no handlers.")

        self.main_logger.handlers[1].filters[0].SetDigits(epoch, iteration)