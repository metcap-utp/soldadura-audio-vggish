import sys

sys.path.append("/home/luis/PlotNeuralNet/")
from pycore.tikzeng import *


def to_TDNN(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=30,
    depth=30,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {RightBandedBox={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_Res2Net(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=35,
    depth=35,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {RightBandedBox={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\ConvColor,
        bandfill=\ConvReluColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_MFA(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=4,
    height=40,
    depth=40,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\FcColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_ASP(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=3,
    height=35,
    depth=35,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\PoolColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_FC(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2,
    height=20,
    depth=20,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\FcColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_BN(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2.2,
    height=28,
    depth=28,
    caption="BatchNorm",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\FcColor,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


def to_Head(
    name,
    offset="(0,0,0)",
    to="(0,0,0)",
    width=2.8,
    height=10,
    depth=10,
    opacity=0.9,
    caption=" ",
):
    return (
        r"""
\pic[shift={"""
        + offset
        + """}] at """
        + to
        + """ 
    {Box={
        name="""
        + name
        + """,
        caption="""
        + caption
        + """,
        fill=\SoftmaxColor,
        opacity="""
        + str(opacity)
        + """,
        height="""
        + str(height)
        + """,
        width="""
        + str(width)
        + """,
        depth="""
        + str(depth)
        + """
        }
    };
"""
    )


arch = [
    to_head("/home/luis/PlotNeuralNet/"),
    to_cor(),
    to_begin(),
    # VGGish: extractor pre-entrenado
    r"""\pic[shift={(0,0,0)}] at (0,0,0)
    {Box={
        name=vggish,
        caption={\parbox{2.4cm}{\centering\small\textbf{VGGish}\\\footnotesize 1$\times$1$\times$128}},
        fill=\ConvColor,
        height=35,
        width=3,
        depth=35
        }
    };""",
    # BatchNorm(128)
    to_BN(
        name="bn_input",
        offset="(2.5,0,0)",
        to="(vggish-east)",
        width=2.2,
        height=28,
        depth=28,
        caption=r"{\parbox{2.0cm}{\centering\small\textbf{BatchNorm}\\\footnotesize 128}}",
    ),
    to_connection("vggish", "bn_input"),
    # TDNN: Conv1d(128, 512, k=5, d=1) + BN + ReLU
    to_TDNN(
        name="tdnn",
        offset="(2.5,0,0)",
        to="(bn_input-east)",
        width=3,
        height=32,
        depth=32,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{TDNN}\\\footnotesize 512$\times$128$\times$5\\\footnotesize d=1}}",
    ),
    to_connection("bn_input", "tdnn"),
    # Res2Net-1: 512, k=3, s=8, d=2
    to_Res2Net(
        name="res2net1",
        offset="(2.5,0,0)",
        to="(tdnn-east)",
        width=3,
        height=35,
        depth=35,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{Res2Net-1}\\\footnotesize 512$\times$512$\times$3\\\footnotesize s=8, d=2}}",
    ),
    to_connection("tdnn", "res2net1"),
    # Res2Net-2: 512, k=3, s=8, d=3
    to_Res2Net(
        name="res2net2",
        offset="(2.5,0,0)",
        to="(res2net1-east)",
        width=3,
        height=35,
        depth=35,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{Res2Net-2}\\\footnotesize 512$\times$512$\times$3\\\footnotesize s=8, d=3}}",
    ),
    to_connection("res2net1", "res2net2"),
    # Res2Net-3: 512, k=3, s=8, d=4
    to_Res2Net(
        name="res2net3",
        offset="(2.5,0,0)",
        to="(res2net2-east)",
        width=3,
        height=35,
        depth=35,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{Res2Net-3}\\\footnotesize 512$\times$512$\times$3\\\footnotesize s=8, d=4}}",
    ),
    to_connection("res2net2", "res2net3"),
    # MFA: Conv 1×1, 1536→1536
    to_MFA(
        name="mfa",
        offset="(3.0,0,0)",
        to="(res2net3-east)",
        width=4,
        height=40,
        depth=40,
        caption=r"{\parbox{3.0cm}{\centering\small\textbf{MFA}\\\footnotesize Conv 1$\times$1\\\footnotesize 1536$\times$1536}}",
    ),
    to_connection("res2net3", "mfa"),
    # ASP: Attentive Stats Pooling, 3072
    to_ASP(
        name="asp",
        offset="(2.8,0,0)",
        to="(mfa-east)",
        width=3,
        height=35,
        depth=35,
        caption=r"{\parbox{2.8cm}{\centering\small\textbf{ASP}\\\footnotesize Attentive Stats\\\footnotesize 3072}}",
    ),
    to_connection("mfa", "asp"),
    # FC + BN: 3072→192
    to_FC(
        name="fc_embedding",
        offset="(2.5,0,0)",
        to="(asp-east)",
        width=2.5,
        height=22,
        depth=22,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{FC + BN}\\\footnotesize 3072$\times$192}}",
    ),
    to_connection("asp", "fc_embedding"),
    # Head: Espesor (192, 3)
    to_Head(
        name="head_espesor",
        offset="(4.0,4.5,0)",
        to="(fc_embedding-east)",
        width=2.8,
        height=10,
        depth=10,
        opacity=0.9,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{Espesor}\\\footnotesize 192$\times$3}}",
    ),
    # Head: Electrodo (192, 4)
    to_Head(
        name="head_electrodo",
        offset="(4.0,0,0)",
        to="(fc_embedding-east)",
        width=2.8,
        height=12,
        depth=12,
        opacity=0.9,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{Electrodo}\\\footnotesize 192$\times$4}}",
    ),
    # Head: Corriente (192, 2)
    to_Head(
        name="head_corriente",
        offset="(4.0,-4.5,0)",
        to="(fc_embedding-east)",
        width=2.8,
        height=8,
        depth=8,
        opacity=0.9,
        caption=r"{\parbox{2.4cm}{\centering\small\textbf{Corriente}\\\footnotesize 192$\times$2}}",
    ),
    # Conexiones a los heads
    r"""\draw [connection]  (fc_embedding-east) -- node {\midarrow} (head_espesor-west);""",
    r"""\draw [connection]  (fc_embedding-east) -- node {\midarrow} (head_electrodo-west);""",
    r"""\draw [connection]  (fc_embedding-east) -- node {\midarrow} (head_corriente-west);""",
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")
    print(f"Archivo generado: {namefile}.tex")
    print(
        f"Para compilar: cd {'/'.join(namefile.split('/')[:-1])} && pdflatex {namefile.split('/')[-1]}.tex"
    )


if __name__ == "__main__":
    main()
