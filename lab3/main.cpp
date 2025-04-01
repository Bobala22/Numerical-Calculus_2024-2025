#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>

#define MATRIX_NUM "5"

#define MATRIX_FILE_NAME "matrici_rare/a_" MATRIX_NUM ".txt"
#define VECTOR_FILE_NAME "vectori_rari/b_" MATRIX_NUM ".txt"
#define OUTPUT_FILE_NAME_REPR_1_ONE_VECTOR "solutions/solution_" MATRIX_NUM "_repr_1_one_vector.txt"
#define OUTPUT_FILE_NAME_REPR_1_TWO_VECTORS "solutions/solution_" MATRIX_NUM "_repr_1_two_vectors.txt"
#define OUTPUT_FILE_NAME_REPR_2_ONE_VECTOR "solutions/solution_" MATRIX_NUM "_repr_2_one_vector.txt"
#define OUTPUT_FILE_NAME_REPR_2_TWO_VECTORS "solutions/solution_" MATRIX_NUM "_repr_2_two_vectors.txt"

using namespace std;

std::vector<double> readVectorFromFile(const std::string &filename)
{
    std::vector<double> result;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return result;
    }

    std::string line;

    if (getline(file, line))
    {
        int n = std::stoi(line);
        result.reserve(n);
    }

    while (getline(file, line))
    {
        if (!line.empty())
        {
            result.push_back(std::stod(line));
        }
    }

    file.close();
    return result;
}

// Prima schema de memorare economica a matricelor (ca in pdf)
// Folosim d = vectorul diagonala si lines = vecotrul de vectori memorati rar
struct elem
{
    double value;
    int poz;
};

bool existsElement(vector<elem> v, int poz)
{
    for (elem e : v)
    {
        if (e.poz == poz)
        {
            return true;
        }
    }
    return false;
}

bool checkNullElementInD(vector<double> d)
{
    for (double elem : d)
    {
        if (elem == 0.0)
        {
            return true;
        }
    }
    return false;
}

// Functie pentru calculul normei euclidiene
double calculateNorm(const vector<double> &x, const vector<double> &y)
{
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); i++)
    {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return sqrt(sum);
}

// Functie pentru metoda Gauss-Seidel cu prima reprezentare (d si lines)
vector<double> gaussSeidelFirstRepr(const vector<double> &d, const vector<vector<elem>> &lines,
                                    const vector<double> &b, double epsilon = 1e-10, int kmax = 10000)
{
    int n = d.size();
    vector<double> xc(n, 0.0);
    vector<double> xp(n, 0.0);
    int k = 0;
    double delta_x;

    do
    {
        xp = xc;

        // Gauss-Seidel
        for (int i = 0; i < n; i++)
        {
            double diag = d[i];
            double suma = 0.0;

            if (i < lines.size())
            {
                for (const elem &e : lines[i])
                {
                    suma += e.value * xc[e.poz];
                }
            }

            if (abs(diag) < 1e-15)
            {
                cerr << "Error: Zero diagonal element at row " << i << endl;
                return vector<double>(n, 0.0);
            }

            // x_i = (b_i - suma(a_ij * x_j)) / a_ii
            xc[i] = (b[i] - suma) / diag;
        }

        delta_x = calculateNorm(xc, xp);
        k++;

        if (k % 100 == 0)
        {
            cout << "Iterația " << k << ", eroare = " << delta_x << endl;
        }
    } while (delta_x >= epsilon && k <= kmax && delta_x <= 1e8);

    cout << "\nMetoda Gauss-Seidel (prima reprezentare - 2 vectori) s-a incheiat dupa " << k << " iteratii.\n";

    if (delta_x < epsilon)
    {
        cout << "Convergenta! Eroare finala: " << delta_x << endl;
    }
    else
    {
        cout << "Divergenta sau precizie insuficienta dupa " << k << " iteratii.\n";
        cout << "Eroare finala: " << delta_x << endl;
    }

    return xc;
}

// Functie pentru metoda Gauss-Seidel cu a doua reprezentare (valori, ind_col, inceput_linii)
vector<double> gaussSeidelSecondRepr(const vector<double> &valori, const vector<int> &ind_col,
                                     const vector<int> &inceput_linii, const vector<double> &b,
                                     double epsilon = 1e-10, int kmax = 10000)
{
    int n = inceput_linii.size() - 1;
    vector<double> xc(n, 0.0);
    vector<double> xp(n, 0.0);
    int k = 0;
    double delta_x;

    do
    {
        xp = xc;

        // Gauss-Seidel
        for (int i = 0; i < n; i++)
        {
            double diag = 0.0;
            double suma = 0.0;

            for (int j = inceput_linii[i]; j < inceput_linii[i + 1]; j++)
            {
                int col = ind_col[j];

                if (col == i)
                {
                    diag = valori[j];
                }
                else
                {
                    suma += valori[j] * xc[col];
                }
            }

            if (abs(diag) < 1e-15)
            {
                cerr << "Error: Zero diagonal element at row " << i << endl;
                return vector<double>(n, 0.0);
            }

            // x_i = (b_i - suma(a_ij * x_j)) / a_ii
            xc[i] = (b[i] - suma) / diag;
        }

        delta_x = calculateNorm(xc, xp);
        k++;

        if (k % 100 == 0)
        {
            cout << "Iterația " << k << ", eroare = " << delta_x << endl;
        }
    } while (delta_x >= epsilon && k <= kmax && delta_x <= 1e8);

    cout << "\nMetoda Gauss-Seidel (a doua reprezentare - 2 vectori) s-a incehiat dupa " << k << " iteratii.\n";

    if (delta_x < epsilon)
    {
        cout << "Convergenta! Eroare finala: " << delta_x << endl;
    }
    else
    {
        cout << "Divergenta sau precizie insuficienta dupa " << k << " iteratii.\n";
        cout << "Eroare finala: " << delta_x << endl;
    }

    return xc;
}

// Calculate the infinity norm of a vector
double infinityNorm(const vector<double> &v)
{
    double maxVal = 0.0;
    for (double val : v)
    {
        maxVal = max(maxVal, abs(val));
    }
    return maxVal;
}

// Multiply matrix (first representation) by vector
vector<double> multiplyFirstRepr(const vector<double> &d, const vector<vector<elem>> &lines,
                                 const vector<double> &x)
{
    int n = d.size();
    vector<double> result(n, 0.0);

    for (int i = 0; i < n; i++)
    {
        // Diagonal element
        result[i] += d[i] * x[i];

        // Non-diagonal elements
        if (i < lines.size())
        {
            for (const elem &e : lines[i])
            {
                result[i] += e.value * x[e.poz];
            }
        }
    }

    return result;
}

// Multiply matrix (second representation) by vector
vector<double> multiplySecondRepr(const vector<double> &valori, const vector<int> &ind_col,
                                  const vector<int> &inceput_linii, const vector<double> &x)
{
    int n = inceput_linii.size() - 1;
    vector<double> result(n, 0.0);

    for (int i = 0; i < n; i++)
    {
        for (int j = inceput_linii[i]; j < inceput_linii[i + 1]; j++)
        {
            int col = ind_col[j];
            result[i] += valori[j] * x[col];
        }
    }

    return result;
}

// Functie pentru metoda Gauss-Seidel cu prima reprezentare (d si lines) - varianta cu un singur vector
vector<double> gaussSeidelFirstReprOneVector(const vector<double> &d, const vector<vector<elem>> &lines,
                                             const vector<double> &b, double epsilon = 1e-10, int kmax = 10000)
{
    int n = d.size();
    vector<double> x_gs(n, 0.0);
    int k = 0;
    double delta_x;

    do
    {
        double sum_squares = 0.0;

        // Gauss-Seidel
        for (int i = 0; i < n; i++)
        {
            double diag = d[i];
            double suma = 0.0;

            if (i < lines.size())
            {
                for (const elem &e : lines[i])
                {
                    suma += e.value * x_gs[e.poz];
                }
            }

            if (abs(diag) < 1e-15)
            {
                cerr << "Error: Zero diagonal element at row " << i << endl;
                return vector<double>(n, 0.0);
            }

            // Salvăm valoarea veche înainte de actualizare
            double old_value = x_gs[i];

            // x_i = (b_i - suma(a_ij * x_j)) / a_ii
            x_gs[i] = (b[i] - suma) / diag;

            // Calculăm pătratul diferenței și îl adăugăm la suma totală
            double diff = x_gs[i] - old_value;
            sum_squares += diff * diff;
        }

        // Calculăm norma euclidiană (rădăcina pătrată din suma pătratelor)
        delta_x = sqrt(sum_squares);

        k++;

        if (k % 100 == 0)
        {
            cout << "Iterația " << k << ", eroare = " << delta_x << endl;
        }
    } while (delta_x >= epsilon && k <= kmax && delta_x <= 1e8);

    cout << "\nMetoda Gauss-Seidel (prima reprezentare - un singur vector) s-a incheiat dupa " << k << " iteratii.\n";

    if (delta_x < epsilon)
    {
        cout << "Convergenta! Eroare finala: " << delta_x << endl;
    }
    else
    {
        cout << "Divergenta sau precizie insuficienta dupa " << k << " iteratii.\n";
        cout << "Eroare finala: " << delta_x << endl;
    }

    return x_gs;
}

// Functie pentru metoda Gauss-Seidel cu a doua reprezentare - varianta cu un singur vector
vector<double> gaussSeidelSecondReprOneVector(const vector<double> &valori, const vector<int> &ind_col,
                                              const vector<int> &inceput_linii, const vector<double> &b,
                                              double epsilon = 1e-10, int kmax = 10000)
{
    int n = inceput_linii.size() - 1;
    vector<double> x_gs(n, 0.0);
    int k = 0;
    double delta_x;

    do
    {
        double sum_squares = 0.0;

        // Gauss-Seidel
        for (int i = 0; i < n; i++)
        {
            double diag = 0.0;
            double suma = 0.0;

            for (int j = inceput_linii[i]; j < inceput_linii[i + 1]; j++)
            {
                int col = ind_col[j];

                if (col == i)
                {
                    diag = valori[j];
                }
                else
                {
                    suma += valori[j] * x_gs[col];
                }
            }

            if (abs(diag) < 1e-15)
            {
                cerr << "Error: Zero diagonal element at row " << i << endl;
                return vector<double>(n, 0.0);
            }

            double old_value = x_gs[i];

            // x_i = (b_i - suma(a_ij * x_j)) / a_ii
            x_gs[i] = (b[i] - suma) / diag;

            double diff = x_gs[i] - old_value;
            sum_squares += diff * diff;
        }

        delta_x = sqrt(sum_squares);

        k++;

        if (k % 100 == 0)
        {
            cout << "Iterația " << k << ", eroare = " << delta_x << endl;
        }
    } while (delta_x >= epsilon && k <= kmax && delta_x <= 1e8);

    cout << "\nMetoda Gauss-Seidel (a doua reprezentare - un singur vector) s-a incheiat dupa " << k << " iteratii.\n";

    if (delta_x < epsilon)
    {
        cout << "Convergenta! Eroare finala: " << delta_x << endl;
    }
    else
    {
        cout << "Divergenta sau precizie insuficienta dupa " << k << " iteratii.\n";
        cout << "Eroare finala: " << delta_x << endl;
    }

    return x_gs;
}

int main()
{
    // A doua schema de memorare economica a matricelor rare (memorarea comprimata pe linii)
    // Folosim 3 vectori: valori = memo elem nenule ale matricei in ordinea liniilor
    // ind_col = stocam indicii coloana a elementelor din "valori"
    // inceput_linii se stochează indicele/poziţia în vectorul valori / ind_col al/a primului element de pe linia i memorat în vectorii valori / ind_col.
    vector<double> valori;
    vector<int> ind_col;
    vector<int> inceput_linii;

    ifstream inputFile(MATRIX_FILE_NAME);
    if (!inputFile)
    {
        cerr << "Error opening file!" << endl;
        return 1;
    }

    std::vector<double> b = readVectorFromFile(VECTOR_FILE_NAME);

    string firstLine;
    int n = 0;
    if (getline(inputFile, firstLine))
    {
        n = stoi(firstLine);
    }

    valori.clear();
    ind_col.clear();
    inceput_linii.resize(n + 1); // +1 for easier iteration (n is matrix dimension)

    // Initialize all positions in inceput_linii to -1 (no elements yet)
    for (int i = 0; i <= n; i++)
    {
        inceput_linii[i] = -1;
    }

    vector<double> d;
    vector<vector<elem>> lines;

    string line;
    int currentLine = -1;
    int positionInValori = 0;

    while (getline(inputFile, line))
    {
        vector<double> elements;
        stringstream ss(line);
        double number;
        while (ss >> number)
        {
            elements.push_back(number);
            while (ss.peek() == ',' || ss.peek() == ' ')
            {
                ss.ignore();
            }
        }
        // -----------------  Prima memorare ----------------------------------
        if (elements[1] == elements[2])
        {
            if (d.size() <= elements[1])
            {
                d.resize(elements[1] + 1);
            }

            if (d[elements[1]] == NULL)
            {
                d.insert(d.begin() + elements[1], elements[0]);
            }
            else
            {
                d[elements[1]] += elements[0];
            }
        }
        else
        {
            elem aux = {elements[0], static_cast<int>(elements[2])};
            if (lines.size() <= elements[1])
            {
                lines.resize(elements[1] + 1);
            }

            if (lines[elements[1]].empty())
            {
                vector<elem> newVector;
                lines.insert(lines.begin() + elements[1], newVector);
                lines[elements[1]].push_back(aux);
            }
            else
            {
                if (existsElement(lines[elements[1]], elements[2]))
                {
                    for (elem &e : lines[elements[1]])
                    {
                        if (e.poz == elements[2])
                        {
                            e.value += elements[0];
                            break; // Found and updated, exit loop
                        }
                    }
                }
                else
                {
                    lines[elements[1]].push_back(aux);
                }
            }
        }
        // ---------------------------------------------------------------------
        // -----------------  A doua memorare ----------------------------------
        int rowIndex = static_cast<int>(elements[1]);
        if (rowIndex > currentLine)
        {
            for (int i = currentLine + 1; i <= rowIndex; i++)
            {
                inceput_linii[i] = positionInValori;
            }
            currentLine = rowIndex;
        }

        // Add the element value to valori and its column to ind_col
        valori.push_back(elements[0]);
        ind_col.push_back(static_cast<int>(elements[2]));
        positionInValori++;
    }
    inputFile.close();

    for (int i = currentLine + 1; i <= n; i++)
    {
        inceput_linii[i] = positionInValori;
    }

    d.resize(d.size() - 1);
    lines.resize(lines.size() - 1);

    if (checkNullElementInD(d))
    {
        throw("There are null elements in d!");
    }

    // Check for dir
    filesystem::path solutionsDir("solutions");
    if (!filesystem::exists(solutionsDir))
    {
        filesystem::create_directory(solutionsDir);
    }

    // Sol with first representation 2 vectors
    vector<double> solutionFirstRepr = gaussSeidelFirstRepr(d, lines, b);

    ofstream outputFileFirstRepr(OUTPUT_FILE_NAME_REPR_1_TWO_VECTORS, ios::out | ios::trunc);
    if (!outputFileFirstRepr)
    {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    for (size_t i = 0; i < solutionFirstRepr.size(); i++)
    {
        outputFileFirstRepr << "x[" << i << "] = " << solutionFirstRepr[i] << endl;
    }

    outputFileFirstRepr.close();

    // Sol with first representation 1 vector
    vector<double> solutionFirstReprOneVector = gaussSeidelFirstReprOneVector(d, lines, b);

    ofstream outputFileFirstReprOneVector(OUTPUT_FILE_NAME_REPR_1_ONE_VECTOR, ios::out | ios::trunc);
    if (!outputFileFirstReprOneVector)
    {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    for (size_t i = 0; i < solutionFirstReprOneVector.size(); i++)
    {
        outputFileFirstReprOneVector << "x[" << i << "] = " << solutionFirstReprOneVector[i] << endl;
    }
    outputFileFirstReprOneVector.close();

    // Sol with second representation with 2 vectors
    vector<double> solution = gaussSeidelSecondRepr(valori, ind_col, inceput_linii, b);

    ofstream outputFileSecondRepr(OUTPUT_FILE_NAME_REPR_2_TWO_VECTORS, ios::out | ios::trunc);
    if (!outputFileSecondRepr)
    {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    for (size_t i = 0; i < solution.size(); i++)
    {
        outputFileSecondRepr << "x[" << i << "] = " << solution[i] << endl;
    }

    outputFileSecondRepr.close();

    // Sol with second representation with 1 vector
    vector<double> solutionSecondReprOneVector = gaussSeidelSecondReprOneVector(valori, ind_col, inceput_linii, b);

    ofstream outputFileSecondReprOneVector(OUTPUT_FILE_NAME_REPR_2_ONE_VECTOR, ios::out | ios::trunc);
    if (!outputFileSecondReprOneVector)
    {
        cerr << "Error opening output file!" << endl;
        return 1;
    }

    for (size_t i = 0; i < solutionSecondReprOneVector.size(); i++)
    {
        outputFileSecondReprOneVector << "x[" << i << "] = " << solutionSecondReprOneVector[i] << endl;
    }
    outputFileSecondReprOneVector.close();

    cout << "\nVerification for first representation solution:" << endl;
    vector<double> Ax1 = multiplyFirstRepr(d, lines, solutionFirstRepr);
    vector<double> residual1(b.size());
    for (size_t i = 0; i < b.size(); i++)
    {
        residual1[i] = Ax1[i] - b[i];
    }
    double residualNorm1 = infinityNorm(residual1);
    cout << "||Ax_gs-b||inf = " << residualNorm1 << endl;

    cout << "\nVerification for second representation solution:" << endl;
    vector<double> Ax2 = multiplySecondRepr(valori, ind_col, inceput_linii, solution);
    vector<double> residual2(b.size());
    for (size_t i = 0; i < b.size(); i++)
    {
        residual2[i] = Ax2[i] - b[i];
    }
    double residualNorm2 = infinityNorm(residual2);
    cout << "||Ax_gs-b||inf = " << residualNorm2 << endl;

    cout << "\nDifference between residual norms: " << abs(residualNorm1 - residualNorm2) << endl;

    return 0;
}

/*
Explicatie pentru divergenta fisierului 5:
In fisierul 5, matricea nu e diagonala dominanta, deci nu se poate aplica metoda Gauss-Seidel.
Pentru ca matricea A sa fie diagonala dominanta, trebuie ca pentru fiecare linie i, |a_ii| > suma(|a_ij|) pentru j de la 1 la n, j != i (n = dimensiunea matricei).
*/