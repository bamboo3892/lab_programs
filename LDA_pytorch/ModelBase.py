from abc import ABCMeta, abstractmethod
import torch
import openpyxl


class LDABase(metaclass=ABCMeta):

    @abstractmethod
    def step(self, subsample_size, parameter_update=False):
        pass


    @abstractmethod
    def log_probability(self, testset=None):
        pass


    @abstractmethod
    def perplexity(self, testset):
        pass


    def summary(self, summary_args):
        """
        summary_args:
            summary_path
            full_docs
            morphome_key
        """

        # torch.save(self, summary_args.summary_path.joinpath("model.pickle"))

        self._summary_print(summary_args)
        self._sammary_wordcloud(summary_args)

        # result.xlsx
        wb = openpyxl.Workbook()
        tmp_ws = wb[wb.get_sheet_names()[0]]
        self._summary_to_excel(summary_args, wb)
        wb.remove_sheet(tmp_ws)
        wb.save(summary_args.summary_path.joinpath("result.xlsx"))


    def _summary_print(self, summary_args):
        pass


    def _sammary_wordcloud(self, summary_args):
        pass


    def _summary_to_excel(self, summary_args, wb):
        pass
