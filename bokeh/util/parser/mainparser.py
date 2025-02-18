from argparse import ArgumentParser

class MainParser(ArgumentParser):
    def __init__(self):
        super().__init__()

        # テスト
        self.add_argument(
            "--test", "-t",
            action="store_true",
            default=False,
            help="[flag] テスト"
        )
        # 必須
        # mode
        self.add_argument(
            "mode",
            type=int,
            default=0,
            help="[0:train 1:test] 学習と推論の選択を行う引数 "
        )
        # コンフィグのパス
        self.add_argument(
            "cfg_path",
            type=str,
            help="[string] コンフィグへのパス"
        )
        
        # 任意
        # 実行環境
        self.add_argument(
            "--device", "-d",
            type=str,
            default="cpu",
            help="[cuda, cpu] プログラムを実行するプロセッサ"
        )

if __name__ == "__main__":
    parser = MainParser()

    args = parser.parse_args()

    print(args.test)
    print(args.device)
