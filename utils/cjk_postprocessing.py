class CJKPostprocessing:
    def __init__(self, enabled) -> None:
        # Define translation dictionary for English to Chinese punctuations
        self.TRANSLATION_TEXT = {
            ".": "。",
            ",": "，",
            "!": "！",
            "?": "？",
            ";": "；",
            ":": "：",
            "(": "（",
            ")": "）",
            "'": "‘",
            "-": "－",
            "--": "——",
            "...": "……",
        }
        self._enabled = enabled
        self._need_conversion = False
        self._in_quotation = False

        import opencc
        self._zh_converter = opencc.OpenCC('t2s')

    def run(self, text: str):
        if not self._enabled:
            return text

        from langdetect import detect, LangDetectException

        if self._need_conversion:
            if text in self.TRANSLATION_TEXT.keys():
                text = self.TRANSLATION_TEXT[text]
                return text

            if '\"' == text:
                self._in_quotation = not self._in_quotation
                text = '“' if self._in_quotation else '”'
                return text

            if '\n' in text:
                self._in_quotation = False

        try:
            if text and not text.isspace():
                if text.isdigit():
                    self._need_conversion = False
                else:
                    self._need_conversion = detect(text) in ['zh-cn', 'zh-tw', 'ko', 'jp']
        except LangDetectException:
            pass  # Ignore failed language detection

        if self._need_conversion:
            text = self._zh_converter.convert(text)

        return text
