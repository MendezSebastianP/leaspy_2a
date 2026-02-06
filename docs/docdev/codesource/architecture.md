# Architecture & Data Flow

This section provides a simplified explanation of Leaspy's internal architecture from a code perspective. We will start with a high-level diagram to give you a global overview and a starting point. Then, we will break it down part by part with more details. By the end, you will understand the architecture, how it works, and how the different functions interact.

Key steps include:
1.  **Data Preparation**: Adapting raw data and putting it into the `Data` format.
2.  **Model Fitting**: Creating a model (e.g. `model = LogisticModel(...)`) and fitting it (`model.fit(...)`).
3.  **Personalization** (Optional): Estimating individual trajectories for each patient (`model.personalize(...)`).

This guide offers two levels of depth: a global explanation and a detailed deep-dive. Feel free to explore according to your needs.

## High-Level Overview

When you run a method, a lot happens under the hood. Here is a simplified code snippet that runs a `LogisticModel`:

```python
from leaspy.io.data import Data
from leaspy.models import LogisticModel

data = Data.from_dataframe(alzheimer_df) # Data part
model = LogisticModel(name="test-model", 
	source_dimension=2)					# Model creation
model.fit(data, "mcmc_saem", seed=42,
    n_iter=100,progress_bar=False)		# Fitting
```

Inside the `leaspy` library, most of the code you interact with is organized into **modules**, which contain **classes**. A class bundles **methods** (functions attached to the class) and **attributes** (data stored on the object). For example, the `LogisticModel.py` module defines two classes, each providing its own methods and attributes.

> If you’re a bit lost, you can check what a [module](https://docs.python.org/3/glossary.html#term-module), [class](https://docs.python.org/3/glossary.html#term-class), [method](https://docs.python.org/3/glossary.html#term-method), and [attribute](https://docs.python.org/3/glossary.html#term-attribute) are.

You can visualize the structure of the `logistic` (*inside /src/leaspy/models/logistic.py*) module like this:

```{mermaid}
%%{init: {"flowchart": {"rankSpacing": 10, "nodeSpacing": 20}} }%%
flowchart TD
    %% Softer cohesive palette (indigo/lilac/teal/sand)
    classDef module fill:#EEF2FF,stroke:#4F46E5,stroke-width:2px,color:#1F2A5A,rx:8,ry:8;
    classDef cls    fill:#F3E8FF,stroke:#7C3AED,stroke-width:2px,color:#3B0764,rx:8,ry:8;
    classDef method fill:#E6FFFB,stroke:#0F766E,stroke-width:1px,color:#134E4A,rx:6,ry:6;
    classDef attr   fill:#FFF7ED,stroke:#C2410C,stroke-width:1px,color:#7C2D12,rx:6,ry:6;

    %% Micro spacer (nearly zero height)
    classDef micro fill:transparent,stroke:transparent,color:transparent,font-size:1px;

    %% Module Subgraph
    subgraph Module["__Module: logistic(.py)__"]
        direction TB

        %% tiny spacer so the subgraph title doesn't get overlapped
        ModuleMicro["."]:::micro

        %% Class 1: Mixin (no pad inside -> less empty space)
        subgraph ClassMixin["__Class: LogisticInitializationMixin__"]
            direction TB
            Method1("Method: _compute_initial_values_for_model_parameters"):::method
        end

        %% Class 2: Model
        subgraph ClassModel["__Class: LogisticModel__"]
            direction TB
            Attr1("Attribute: name"):::attr
            Method2("Method: \__init__"):::method
            Method3("Method: get_variables_specs"):::method
            Method4("Method: metric"):::method
            Method5("Method: model_with_sources"):::method

            %% Force vertical listing inside the class
            Attr1 --> Method2 --> Method3 --> Method4 --> Method5
        end

        %% Force vertical stacking of the two class subgraphs
        ClassMixin --> ClassModel

        %% Anchor the micro node so it actually affects layout
        ModuleMicro --> ClassMixin
    end

    %% Apply Classes
    Module:::module
    ClassMixin:::cls
    ClassModel:::cls

    %% Hide all layout-forcing edges (keeps it clean)
    linkStyle default stroke:transparent,stroke-width:0;
```

This example allows us to see how is structured a module, we will go deeper in the other modules that compose the worflow from `leaspy`, specially in the most the most simple scenario: a logistic regression.

## Simplified workflow structure

When you create your model and you fit it a lot happens under the hood. For instance `LogisticModel` inherits methods and attributes from other classes in a chain. `LogisticModel` inherits from `RiemanianManifoldModel`, which inherits from other classes, and so on.

Here is the inheritance chain for the Logistic model. You can click on the nodes to see the details of each class.

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '18px', 'fontFamily': 'arial', 'lineWidth': '3px' }}}%%
flowchart TD
    %% Styling Definitions
    classDef interface fill:#E0F2F1,stroke:#004D40,stroke-width:2px,color:#004D40,rx:5,ry:5;
    classDef base fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C,rx:5,ry:5;
    classDef mixin fill:#F3E5F5,stroke:#4A148C,stroke-width:2px,color:#4A148C,rx:5,ry:5;
    classDef impl fill:#E3F2FD,stroke:#0D47A1,stroke-width:3px,color:#0D47A1,rx:5,ry:5;

    %% Nodes
    Interface["ModelInterface"]:::interface
    Base["BaseModel"]:::base
    Stateful["StatefulModel"]:::base
    SAEM["McmcSaemCompatibleModel"]:::base
    Time["TimeReparametrizedModel"]:::base
    Rieman["RiemanianManifoldModel"]:::base
    Mixin["LogisticInitializationMixin"]:::mixin
    Logistic["LogisticModel"]:::impl

    %% Links
    Interface --> Base
    Base --> Stateful
    Stateful --> SAEM
    SAEM --> Time
    Time --> Rieman
    Rieman --> Logistic
    Mixin --> Logistic

    %% Click Interactions (Links to module documentation)
    click Interface "logistic/ModelInterface.html" "Go to ModelInterface"
    click Base "logistic/BaseModel.html" "Go to BaseModel"
    click Stateful "logistic/StatefulModel.html" "Go to StatefulModel"
    click SAEM "logistic/McmcSaemCompatibleModel.html" "Go to McmcSaemCompatibleModel"
    click Time "logistic/TimeReparametrizedModel.html" "Go to TimeReparametrizedModel"
    click Rieman "logistic/RiemanianManifoldModel.html" "Go to RiemanianManifoldModel"
    click Mixin "logistic/LogisticInitializationMixin.html" "Go to LogisticInitializationMixin"
    click Logistic "logistic/LogisticModel.html" "Go to LogisticModel"
```

This diagram is clickable. If you want more details about a specific module, you can click on it. For now, let's start with the base of the inheritance chain: [`BaseModel`](logistic/BaseModel).