class Postprocessing:
    def run(self, text: str):
        return text


class PostprocessingGroup:
    def __init__(self, *args: Postprocessing) -> None:
        self._processors = []
        for arg in args:
            if isinstance(arg, Postprocessing):
                self._processors.append(arg)

    def __call__(self, text: str):
        for processor in self._processors:
            text = processor.run(text)

        return text


class ReasoningPostprocessing(Postprocessing):
    def run(self, text: str):
        if text.startswith("<think>"):
            return "--------------------------------------------------*Reasoning starts*:\n "
        elif text.startswith("</think>"):
            return "--------------------------------------------------*Reasoning ends*.\n\n"
        else:
            return text


class MathPostprocessing(Postprocessing):
    def __init__(self) -> None:
        self._in_eq = False
        self._in_inline_eq = False

    def run(self, text: str):
        if text.strip() == "$$":  # Block equation sign
            self._in_eq = not self._in_eq
            text = '\\[' if self._in_eq else '\\]'
            return text
        elif not self._in_eq and "$$" not in text:  # Not inside a block equation
            if text.strip().startswith("$") or text.strip().endswith("$"):  # Inline equation sign
                self._in_inline_eq = not self._in_inline_eq
                text = text.replace("$", "\\(") if self._in_inline_eq else text.replace("$", "\\)")
                return text

        if "\n" in text:
            self._in_inline_eq = False

        return text


class CJKPostprocessing(Postprocessing):
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
