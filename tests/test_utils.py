import unittest
import sys
sys.path.append('/Users/carsonmckee/Dev/bayesglm_2/bayesglm')
import utils
import pandas

class TestModelMatrix(unittest.TestCase):

    def setUp(self):
        self.data = pandas.DataFrame(data=[[1,"A",3,4,5], [6,"B",8,9,10], [11,"C",13,14,15]],
                                     columns=["response", "x1", "x2", "x3", "x4"])

    def test_all(self):
        expected_mat = self.data.drop(["response", "x1"], axis=1)
        expected_mat["Intercept"] = 1
        expected_mat["x1B"] = (self.data["x1"] == "B").astype(int)
        expected_mat["x1C"] = (self.data["x1"] == "C").astype(int)

        expected_mat = expected_mat.sort_index(axis=1)
        res, mat = utils.model_matrix("response~.", self.data)
        pandas.testing.assert_frame_equal(mat.sort_index(axis=1), expected_mat)
    
    def test_all_minus_x1(self):
        expected_mat = self.data.drop(["response", "x1"], axis=1)
        expected_mat["Intercept"] = 1

        expected_mat = expected_mat.sort_index(axis=1)
        res, mat = utils.model_matrix("response~.-x1", self.data)
        pandas.testing.assert_frame_equal(mat.sort_index(axis=1), expected_mat)

    def test_all_minus_intercept(self):
        expected_mat = self.data.drop(["response","x1"], axis=1)
        expected_mat["x1B"] = (self.data["x1"] == "B").astype(int)
        expected_mat["x1C"] = (self.data["x1"] == "C").astype(int)

        expected_mat = expected_mat.sort_index(axis=1)
        res, mat = utils.model_matrix("response~.-1", self.data)
        pandas.testing.assert_frame_equal(mat.sort_index(axis=1), expected_mat)
    
    def test_simple_add(self):
        expected_mat = self.data.drop(["response","x1","x3"], axis=1)
        expected_mat["Intercept"] = 1
        expected_mat["x1B"] = (self.data["x1"] == "B").astype(int)
        expected_mat["x1C"] = (self.data["x1"] == "C").astype(int)

        expected_mat = expected_mat.sort_index(axis=1)
        res, mat = utils.model_matrix("response~ x1 + x2 + x4", self.data)
        pandas.testing.assert_frame_equal(mat.sort_index(axis=1), expected_mat)
    
    def test_all_and_subtract(self):
        expected_mat = self.data.drop(["response","x1","x4"], axis=1)
        expected_mat["x1B"] = (self.data["x1"] == "B").astype(int)
        expected_mat["x1C"] = (self.data["x1"] == "C").astype(int)

        expected_mat = expected_mat.sort_index(axis=1)
        res, mat = utils.model_matrix("response~. - x4 -1", self.data)
        pandas.testing.assert_frame_equal(mat.sort_index(axis=1), expected_mat)

    def test_simple_interaction(self):
        expected_mat = self.data.drop(["response","x1","x2", "x3", "x4"], axis=1)
        expected_mat["x2*x3"] = self.data["x2"]*self.data["x3"]

        expected_mat = expected_mat.sort_index(axis=1)
        res, mat = utils.model_matrix("response~ x2*x3 -1", self.data)
        pandas.testing.assert_frame_equal(mat.sort_index(axis=1), expected_mat)

    def test_multiple_interaction(self):
        expected_mat = self.data.drop(["response","x1","x2", "x3", "x4"], axis=1)
        expected_mat["x2*x3*x4"] = self.data["x2"]*self.data["x3"]*self.data["x4"]

        expected_mat = expected_mat.sort_index(axis=1)
        res, mat = utils.model_matrix("response~ x2*x3*x4 -1", self.data)
        pandas.testing.assert_frame_equal(mat.sort_index(axis=1), expected_mat)

        res, mat2 = utils.model_matrix("response~ x2*x4*x3 -1", self.data)
        pandas.testing.assert_frame_equal(mat2.sort_index(axis=1), expected_mat)

    def test_all_plus_simple_interaction(self):
        expected_mat = self.data.drop(["response", "x1"], axis=1)
        expected_mat["x2*x3"] = self.data["x2"]*self.data["x3"]
        expected_mat["Intercept"] = 1
        expected_mat["x1B"] = (self.data["x1"] == "B").astype(int)
        expected_mat["x1C"] = (self.data["x1"] == "C").astype(int)

        expected_mat = expected_mat.sort_index(axis=1)
        res, mat = utils.model_matrix("response~ . + x2*x3", self.data)
        pandas.testing.assert_frame_equal(mat.sort_index(axis=1), expected_mat)

        res, mat2 = utils.model_matrix("response~ . + x3*x2", self.data)
        pandas.testing.assert_frame_equal(mat2.sort_index(axis=1), expected_mat)

    def test_all_plus_multi_interaction(self):

        expected_mat = self.data.drop(["response", "x1"], axis=1)
        expected_mat["x2*x3*x4"] = self.data["x2"]*self.data["x3"]*self.data["x4"]
        expected_mat["Intercept"] = 1
        expected_mat["x1B"] = (self.data["x1"] == "B").astype(int)
        expected_mat["x1C"] = (self.data["x1"] == "C").astype(int)

        expected_mat = expected_mat.sort_index(axis=1)
        res, mat = utils.model_matrix("response~ . + x2*x3*x4", self.data)
        pandas.testing.assert_frame_equal(mat.sort_index(axis=1), expected_mat)

        res, mat2 = utils.model_matrix("response~ . + x3*x2*x4", self.data)
        pandas.testing.assert_frame_equal(mat2.sort_index(axis=1), expected_mat)

    def test_discrete_inter_discrete(self):
        self.data["x5"] = ["D", "E", "F"]

        expected_mat = self.data.drop(["response", "x1", "x2", "x3", "x4", "x5"], axis=1)
        expected_mat["Intercept"] = 1
        expected_mat["x1B*x5E"] = (self.data["x1"] == "B").astype(int) * (self.data["x5"] == "E").astype(int)
        expected_mat["x1C*x5E"] = (self.data["x1"] == "C").astype(int) * (self.data["x5"] == "E").astype(int)
        expected_mat["x1B*x5F"] = (self.data["x1"] == "B").astype(int) * (self.data["x5"] == "F").astype(int)
        expected_mat["x1C*x5F"] = (self.data["x1"] == "C").astype(int) * (self.data["x5"] == "F").astype(int)
        expected_mat = expected_mat.sort_index(axis=1)

        res, mat = utils.model_matrix("response~ x1*x5", self.data)
        pandas.testing.assert_frame_equal(mat.sort_index(axis=1), expected_mat)

        res, mat2 = utils.model_matrix("response~ x5*x1", self.data)
        pandas.testing.assert_frame_equal(mat2.sort_index(axis=1), expected_mat)
        
    def test_discrete_inter_cont(self):

        expected_mat = self.data.drop(["response", "x1", "x2", "x3", "x4"], axis=1)
        expected_mat["Intercept"] = 1
        expected_mat["x2*x1B"] = (self.data["x1"] == "B").astype(int) * self.data["x2"]
        expected_mat["x2*x1C"] = (self.data["x1"] == "C").astype(int) * self.data["x2"]
        expected_mat = expected_mat.sort_index(axis=1)

        res, mat = utils.model_matrix("response~ x1*x2", self.data)
        pandas.testing.assert_frame_equal(mat.sort_index(axis=1), expected_mat)

        res, mat2 = utils.model_matrix("response~ x1*x2", self.data)
        pandas.testing.assert_frame_equal(mat2.sort_index(axis=1), expected_mat)


class TestChecks(unittest.TestCase):
    
    def test_check_distribution(self):
        pass


if __name__ == '__main__':
    unittest.main()