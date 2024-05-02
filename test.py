from utils import *

if __name__ == '__main__':
    bias = relative_bias(-13856.372498899807, -14833.9960)
    print(f"PSID (IPTW) relative bias: {bias}")
    bias = relative_bias(-13856.372498899807, -107.1296)
    print(f"PSID (standardization) relative bias: {bias}")
    bias = relative_bias(-13856.372498899807, -15101.4387)
    print(f"PSID (AIPW) relative bias: {bias}")

    bias = (-913.46)/(-13856.372498899807)
    print(f"PSID (IPTW) relative bias - RealCause: {bias}")
    bias = (-435.8)/(-13856.372498899807)
    print(f"PSID (standardization) relative bias - RealCause: {bias}")

    bias = relative_bias(-6977.127761167221, -7385.3878)
    print(f"CPS (IPTW) relative bias: {bias}")
    bias = relative_bias(-6977.127761167221, -14456.9846)
    print(f"CPS (standardization) relative bias: {bias}")
    bias = relative_bias(-6977.127761167221, -11003.2622)
    print(f"CPS (AIPW) relative bias: {bias}")

    bias = (1691.40)/(-6977.127761167221)
    print(f"CPS (IPTW) relative bias - RealCause: {bias}")
    bias = (2131.24)/(-6977.127761167221)
    print(f"CPS (standardization) relative bias - RealCause: {bias}")

    bias = relative_bias(-0.06934245660881175, -0.1545)
    print(f"Twins (IPTW) relative bias: {bias}")
    bias = relative_bias(-0.06934245660881175,  -0.0006)
    print(f"Twins (standardization) relative bias: {bias}")
    bias = relative_bias(-0.06934245660881175,  -0.1337)
    print(f"Twins (AIPW) relative bias: {bias}")

    bias = (-0.06934245660881175)/(-0.06934245660881175)